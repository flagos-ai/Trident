# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Python frontend construction, diagnostics and lexical input caching."""

import unittest

import sympy
import torch
from trident.core import ir
from trident.core.dialects import arith, func, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.importer import TridentFxImporter
from trident.input import InputNodeBuilder, InputTableBuilder
from trident.ir import (
    bind,
    box,
    checked_call,
    constant,
    container_get,
    make_result_types,
    native_constant,
    pure,
)
from trident.ownership import is_value_in_scope

from test.support.ir import new_context


class ImporterTest(unittest.TestCase):
    @staticmethod
    def _extraction_count(module: ir.Module) -> int:
        extractions: list[ir.Operation] = []

        def collect(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "tvm_ffi.array.get":
                extractions.append(operation)
            return ir.WalkResult.ADVANCE

        module.operation.walk(collect)
        return len(extractions)

    def test_bind_arities(self) -> None:
        for input_count, output_count in [(0, 0), (0, 2), (2, 0), (2, 3)]:
            with (
                self.subTest(inputs=input_count, outputs=output_count),
                new_context(),
                ir.Location.unknown(),
            ):
                module = ir.Module.create()
                result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.IntType.get())
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp(
                        "test",
                        ([result_type] * input_count, [result_type] * output_count),
                    )
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    with bind(
                        block.arguments, result_types=[result_type] * output_count
                    ) as operation:
                        self.assertEqual(len(operation.arguments), input_count)
                        values = (
                            pure([constant(7)] * output_count) if output_count else []
                        )
                        tvm_ffi_d.MonadYieldOp(values)

                    self.assertEqual(ir.InsertionPoint.current.block, block)
                    self.assertEqual(len(operation.outputs), output_count)
                    func.ReturnOp(operation.outputs)

                self.assertTrue(module.operation.verify())

    def test_bind_empty_restores_insertion_point_on_exception(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                with (
                    self.assertRaisesRegex(RuntimeError, "stop"),
                    bind([], result_types=[]),
                ):
                    tvm_ffi_d.MonadYieldOp([])

                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, block)
                func.ReturnOp([])

            self.assertTrue(module.operation.verify())

    def test_bind_result_contract(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            module = importer.module
            integer = tvm_ffi_d.IntType.get()
            payload = tvm_ffi_d.ResultType.get(integer)
            unit = tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
            with ir.InsertionPoint(module.body):
                value = native_constant(7)
                count = len(module.body.operations)
                with (
                    self.assertRaisesRegex(AssertionError, "bind inputs"),
                    bind([value], result_types=[payload]),
                ):
                    self.fail("invalid inputs entered the bind body")
                self.assertEqual(len(module.body.operations), count)
                none = constant(None)
                results = pure(
                    [none, none, value, none], result_types=[unit, unit, payload, unit]
                )
                with bind(results, result_types=[payload]) as operation:
                    first, second, argument, last = operation.arguments
                    self.assertEqual(
                        [first.type, second.type, last.type], [unit.payload_type] * 3
                    )
                    self.assertEqual(argument.type, integer)
                    tvm_ffi_d.MonadYieldOp(pure([argument], result_types=[payload]))
            self.assertTrue(module.operation.verify())
            parsed = ir.Module.parse(f"{module}")
            parsed_bind = next(
                op
                for op in parsed.body.operations
                if isinstance(op, tvm_ffi_d.MonadBindOp)
            )
            self.assertEqual(
                [value.type for value in parsed_bind.arguments],
                [unit.payload_type, unit.payload_type, integer, unit.payload_type],
            )

    def test_cache_lives_until_defining_block_exit(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = tvm_ffi_d.TensorType.get()
            array = tvm_ffi_d.ArrayType.get(tensor)
            unit = tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
            builder = InputTableBuilder(
                [("values", InputNodeBuilder(array, [InputNodeBuilder(tensor)]))],
                ["values"],
            )
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([array], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                table = builder.build(block.arguments)
                first = table.extract(("values", 0))
                with bind(first, result_types=[unit]):
                    cached = table.extract(("values", 0))
                    with bind(cached, result_types=[unit]) as reused:
                        tvm_ffi_d.MonadYieldOp(pure([]))

                    tvm_ffi_d.MonadYieldOp(reused.outputs)

                # Ending the Python context does not end the IR block lifetime.
                self.assertTrue(is_value_in_scope(first[0]))
                cached = table.extract(("values", 0))
                with bind(cached, result_types=[unit]):
                    tvm_ffi_d.MonadYieldOp(pure([]))

                func.ReturnOp([])

            self.assertEqual(self._extraction_count(module), 1)
            self.assertTrue(module.operation.verify())

    def test_cache_rebuilds_in_sibling_blocks(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = tvm_ffi_d.TensorType.get()
            array = tvm_ffi_d.ArrayType.get(tensor)
            unit = tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
            builder = InputTableBuilder(
                [("values", InputNodeBuilder(array, [InputNodeBuilder(tensor)]))],
                ["values"],
            )
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([array], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                table = builder.build(block.arguments)
                predicate = arith.constant(ir.IntegerType.get_signless(1), True)
                branch = scf.IfOp(predicate, [], has_else=True)
                for child in (branch.then_block, branch.else_block):
                    with ir.InsertionPoint(child):
                        values = table.extract(("values", 0))
                        with bind(values, result_types=[unit]):
                            tvm_ffi_d.MonadYieldOp(pure([]))

                        scf.YieldOp([])

                func.ReturnOp([])

            self.assertEqual(self._extraction_count(module), 2)
            self.assertTrue(module.operation.verify())

    def test_container_array_operations(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            with ir.InsertionPoint(importer.module.body):
                integer = box(constant(7))
                array_type = tvm_ffi_d.ArrayType.get(integer.type)
                created = tvm_ffi_d.ArrayCreateOp(array_type, [integer]).result
                self.assertEqual(created.type, array_type)
                size = tvm_ffi_d.ArraySizeOp(created).result
                self.assertEqual(size.type, integer.type)
                element = tvm_ffi_d.ArrayGetOp(created, constant(0)).result
                self.assertEqual(element.type, integer.type)
                with bind(
                    pure([created]), result_types=[tvm_ffi_d.ResultType.get(array_type)]
                ) as bound:
                    [array] = bound.arguments
                    for target in (integer.type, ir.IntegerType.get_signless(64)):
                        with self.subTest(target=target):
                            [result] = container_get(array, 0, target)
                            self.assertEqual(result.type.payload_type, integer.type)
                    tvm_ffi_d.MonadYieldOp(pure([array]))
            self.assertTrue(importer.module.operation.verify())

    def test_container_inference(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with (
            context,
            ir.Location.unknown(context),
            ir.InsertionPoint(importer.module.body),
        ):
            for value, expected in (
                ([], "!tvm_ffi.array<!tvm_ffi.any>"),
                ((), "!tvm_ffi.tuple<>"),
                ([1, 2], "!tvm_ffi.array<!tvm_ffi.int>"),
                ([1, None], "!tvm_ffi.array<!tvm_ffi.any>"),
                (
                    ["a", "b"],
                    "!tvm_ffi.array<!tvm_ffi.str>",
                ),
                ([torch.float32, torch.float64], "!tvm_ffi.array<!tvm_ffi.int>"),
                ((torch.float32, 1), "!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.int>"),
                ([[1], [2]], "!tvm_ffi.array<!tvm_ffi.array<!tvm_ffi.int>>"),
            ):
                with self.subTest(value=value):
                    [result] = importer.build_arguments([value])
                    self.assertEqual(f"{result.type.payload_type}", expected)
            self.assertTrue(importer.module.operation.verify())

    def test_container_tuple_operations(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            with ir.InsertionPoint(importer.module.body):
                integer, boolean = box(constant(7)), box(constant(True))
                type_ = tvm_ffi_d.TupleType.get([integer.type, boolean.type])
                tuple_ = tvm_ffi_d.TupleConstructOp(type_, [integer, boolean]).result
                self.assertEqual(tuple_.type, type_)
                elements = tvm_ffi_d.TupleUnconstructOp(type_.types, tuple_).elements
                self.assertEqual(
                    [value.type for value in elements], [integer.type, boolean.type]
                )
                for index, expected in (
                    (0, integer.type),
                    (1, boolean.type),
                    (-1, boolean.type),
                    (-2, integer.type),
                ):
                    with self.subTest(index=index):
                        [result] = container_get(tuple_, index, expected)
                        self.assertEqual(result.type.payload_type, expected)
                for index in (-3, 2):
                    with (
                        self.subTest(index=index),
                        self.assertRaisesRegex(
                            AssertionError, "tuple index out of range"
                        ),
                    ):
                        container_get(tuple_, index, integer.type)
                empty_type = tvm_ffi_d.TupleType.get([])
                empty = tvm_ffi_d.TupleConstructOp(empty_type, []).result
                self.assertEqual(
                    len(tvm_ffi_d.TupleUnconstructOp([], empty).elements), 0
                )
            self.assertTrue(importer.module.operation.verify())
            self.assertTrue(ir.Module.parse(f"{importer.module}").operation.verify())

    def test_empty_arguments_and_explicit_none(self) -> None:
        def collect(operation: ir.Operation) -> list[ir.OpView]:
            calls: list[ir.OpView] = []
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        if isinstance(
                            child, (tvm_ffi_d.ArrayCreateOp, tvm_ffi_d.TupleConstructOp)
                        ):
                            calls.append(child)
                        calls.extend(collect(child.operation))
            return calls

        for arguments, count, array_calls in (
            ([], 1, 0),
            ([None], 1, 0),
            ([[], (), None], 3, 2),
        ):
            with self.subTest(arguments=arguments):
                context = new_context()
                importer = TridentFxImporter(context=context, specialization_id=0)
                with context, ir.Location.unknown(context):
                    none_result = tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
                    with ir.InsertionPoint(importer.module.body):
                        results = importer.build_arguments(arguments)
                        self.assertEqual(len(results), count)
                        with bind(results, result_types=[none_result]) as bound:
                            self.assertEqual(len(bound.arguments), count)
                            self.assertIsInstance(
                                bound.arguments[-1].type, tvm_ffi_d.NoneType
                            )
                            tvm_ffi_d.MonadYieldOp(
                                pure([constant(None)], result_types=[none_result])
                            )
                    self.assertTrue(importer.module.operation.verify())
                    calls = collect(importer.module.operation)
                    self.assertEqual(len(calls), array_calls)
                    for call in calls:
                        self.assertEqual(len(call.elements), 0)

    def test_frontend_uses_semantic_object_types(self) -> None:
        forbidden = (
            tvm_ffi_d.CAnyType,
            tvm_ffi_d.CArrayType,
            tvm_ffi_d.CExceptionType,
            tvm_ffi_d.CFunctionType,
            tvm_ffi_d.CStrType,
            tvm_ffi_d.CTensorType,
            tvm_ffi_d.ObjectType,
        )

        def check_type(type_: ir.Type) -> None:
            self.assertNotIsInstance(type_, forbidden)
            if isinstance(type_, tvm_ffi_d.ArrayType):
                check_type(type_.element_type)
            elif isinstance(type_, tvm_ffi_d.ResultType):
                check_type(type_.payload_type)
            elif isinstance(type_, (tvm_ffi_d.TupleType, tvm_ffi_d.UnionType)):
                for member in type_.types:
                    check_type(member)
            elif isinstance(type_, ir.FunctionType):
                for member in (*type_.inputs, *type_.results):
                    check_type(member)

        def check_attribute(attribute: ir.Attribute) -> None:
            if isinstance(attribute, ir.TypeAttr):
                check_type(attribute.value)
            elif isinstance(attribute, ir.ArrayAttr):
                for member in attribute:
                    check_attribute(member)
            elif isinstance(attribute, ir.DictAttr):
                for member in attribute:
                    check_attribute(member.attr)

        def check_operation(operation: ir.Operation) -> None:
            for value in (*operation.operands, *operation.results):
                check_type(value.type)
            for attribute in operation.attributes:
                check_attribute(
                    operation.attributes[attribute]
                    if isinstance(attribute, str)
                    else attribute.attr
                )
            for region in operation.regions:
                for block in region.blocks:
                    for argument in block.arguments:
                        check_type(argument.type)
                    for child in block.operations:
                        check_operation(child.operation)

        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            tensor = tvm_ffi_d.TensorType.get()
            string = tvm_ffi_d.StrType.get()
            with ir.InsertionPoint(importer.module.body):
                function = func.FuncOp("semantic", ([tensor, string], []))
                with ir.InsertionPoint(function.add_entry_block()):
                    checked_call("test.identity", [function.arguments[0]], tensor)
                    pure([tvm_ffi_d.ExceptionOp("GuardMatch").result])
                    importer.build_arguments(
                        [torch.zeros(2), ("", "1234567", "12345678"), [None, 3]]
                    )
                    func.ReturnOp([])
            self.assertTrue(importer.module.operation.verify())
            self.assertNotIn("tvm_ffi.cleanup", f"{importer.module}")
            self.assertNotIn("tvm_ffi.retain", f"{importer.module}")
            check_operation(importer.module.operation)
            parsed = ir.Module.parse(f"{importer.module}")
            check_operation(parsed.operation)

    def test_pure_inference_and_typed_boundaries(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            integer = ir.IntegerType.get_signless(64)
            boxed = tvm_ffi_d.IntType.get()
            union = tvm_ffi_d.UnionType.get([boxed, tvm_ffi_d.NoneType.get()])
            with ir.InsertionPoint(module.body):
                value = native_constant(7)
                inferred = pure([value])
                self.assertEqual(inferred[0].type.payload_type, boxed)
                self.assertEqual(make_result_types(iter([integer])), [inferred[0].type])
                string = constant("text")
                [string_result] = pure([string])
                self.assertIsInstance(string.type, tvm_ffi_d.StrType)
                self.assertEqual(string_result.type.payload_type, string.type)
                self.assertEqual(make_result_types([string.type]), [string_result.type])
                empty = pure([])
                self.assertIsInstance(empty[0].type.payload_type, tvm_ffi_d.NoneType)
                for target in (boxed, tvm_ffi_d.AnyType.get(), union):
                    with self.subTest(target=target):
                        result_types = make_result_types([target])
                        [result] = pure([value], result_types=result_types)
                        operation = result.owner.opview
                        self.assertEqual(operation.values[0].type, target)
                        widened = operation.values[0]
                        self.assertIsInstance(widened.owner.opview, tvm_ffi_d.UpcastOp)
                        narrowed = tvm_ffi_d.downcast(boxed, widened)
                        self.assertEqual(narrowed.type, boxed)
                        self.assertIsInstance(
                            narrowed.owner.opview, tvm_ffi_d.DowncastOp
                        )
                with self.assertRaisesRegex(AssertionError, "payload count"):
                    pure([value, value], result_types=make_result_types([integer]))
                with self.assertRaisesRegex(AssertionError, "payload type"):
                    pure([value], result_types=make_result_types([ir.F64Type.get()]))
                with (
                    self.assertRaisesRegex(RuntimeError, "construction failed"),
                    bind(inferred, result_types=make_result_types([integer])) as bound,
                ):
                    tvm_ffi_d.MonadYieldOp(pure(bound.arguments))
                    raise RuntimeError("construction failed")
                self.assertEqual(ir.InsertionPoint.current.block, module.body)
                with self.assertRaisesRegex(
                    AssertionError, "outside its defining block"
                ):
                    pure(bound.arguments)
            self.assertTrue(module.operation.verify())

    def test_unsupported_symbolic_power(self) -> None:
        importer = TridentFxImporter(context=new_context(), specialization_id=0)
        base, exponent = sympy.symbols("base exponent")
        for power in (-1, sympy.Rational(1, 2), exponent):
            with self.subTest(exponent=power):
                expression = sympy.Pow(base, power, evaluate=False)
                with self.assertRaisesRegex(
                    NotImplementedError, "unbound symbolic FX expression"
                ):
                    importer.symbolic(expression)
