# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Python frontend construction, diagnostics and uncached input extraction."""

import operator
import unittest

import sympy
import torch
from trident.core import ir
from trident.core.dialects import func, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.importer import TridentFxImporter
from trident.input import InputNodeBuilder, InputTableBuilder
from trident.ir import (
    BINARY_ARITHMETIC,
    BindScope,
    ScopeStack,
    checked_call,
    constant,
    container_get,
    make_result_types,
    native,
    power,
    pure,
)

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

    def test_import_program_packed_boundary(self) -> None:
        class Outputs(torch.nn.Module):
            def __init__(self, count: int) -> None:
                super().__init__()
                self.count = count

            def forward(self, value: torch.Tensor):
                return tuple(value + index for index in range(self.count))

        for count in (0, 1, 2):
            with self.subTest(outputs=count), new_context() as context:
                program = torch.export.export(Outputs(count), (torch.ones(2),))
                importer = TridentFxImporter(context=context, specialization_id=0)
                function = importer.import_program(program, func_name="graph")
                self.assertIsInstance(function, tvm_ffi_d.FuncOp)
                self.assertEqual(function.sym_visibility.value, "private")
                self.assertEqual(
                    list(function.function_type.value.results),
                    [tvm_ffi_d.AnyType.get()],
                )
                self.assertIn("emit_tvm_ffi_abi", function.attributes)
                self.assertTrue(importer.module.operation.verify())
                operations: list[str] = []

                def collect(
                    operation: ir.Operation, names: list[str] = operations
                ) -> ir.WalkResult:
                    names.append(operation.name)
                    return ir.WalkResult.ADVANCE

                function.operation.walk(collect)
                self.assertIn("tvm_ffi.return", operations)
                self.assertIn("tvm_ffi.exception", operations)
                self.assertNotIn("func.return", operations)
                if count > 1:
                    self.assertIn("tvm_ffi.tuple.construct", operations)

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
                    with BindScope(
                        [result_type] * output_count, block.arguments
                    ) as scope:
                        payloads = scope.arguments
                        self.assertEqual(len(payloads), input_count)
                        values = (
                            pure([constant(7)] * output_count) if output_count else []
                        )
                        scope.yield_(values)
                        bound_results = scope.results

                    self.assertEqual(ir.InsertionPoint.current.block, block)
                    self.assertEqual(len(bound_results), output_count)
                    func.return_(bound_results)

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
                    BindScope([], []) as scope,
                ):
                    scope.yield_([])

                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, block)
                func.return_([])

            self.assertTrue(module.operation.verify())

    def test_bind_scopes(self) -> None:
        for count in (0, 1, 3, 1100):
            with self.subTest(count=count), new_context(), ir.Location.unknown():
                module = ir.Module.create()
                types = make_result_types([tvm_ffi_d.IntType.get()])
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp("chain", (types, types))
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    results = block.arguments
                    with ScopeStack() as stack:
                        for _ in range(count):
                            [value] = stack.bind(types, results)
                            results = pure([value])
                        results = stack.yield_(results)
                    self.assertEqual(ir.InsertionPoint.current.block, block)
                    func.return_(results)
                self.assertTrue(module.operation.verify())

        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                with (
                    self.assertRaisesRegex(RuntimeError, "stop"),
                    BindScope(make_result_types([]), pure([])) as outer,
                ):
                    with BindScope(make_result_types([]), pure([])) as inner:
                        inner.yield_(pure([]))
                    outer.yield_(inner.results)
                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, module.body)
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
                value = constant(7)
                count = len(module.body.operations)
                with (
                    self.assertRaisesRegex(AssertionError, "bind inputs"),
                    BindScope([payload], [value]) as scope,
                ):
                    self.fail("invalid inputs entered the bind body")
                self.assertEqual(len(module.body.operations), count)
                none = constant(None)
                results = pure(
                    [none, none, value, none], result_types=[unit, unit, payload, unit]
                )
                with BindScope([payload], results) as scope:
                    payloads = scope.arguments
                    first, second, argument, last = payloads
                    self.assertEqual(
                        [first.type, second.type, last.type], [unit.payload_type] * 3
                    )
                    self.assertEqual(argument.type, integer)
                    scope.yield_(pure([argument], result_types=[payload]))
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

    def test_input_extraction_is_fresh_in_nested_blocks(self) -> None:
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
                second = table.extract(("values", 0))
                self.assertNotEqual(first[0], second[0])
                with BindScope([unit], first) as scope:
                    extracted = table.extract(("values", 0))
                    with BindScope([unit], extracted) as nested:
                        nested.yield_(pure([]))
                        nested_results = nested.results

                    scope.yield_(nested_results)

                extracted = table.extract(("values", 0))
                with BindScope([unit], extracted) as scope:
                    scope.yield_(pure([]))

                func.return_([])

            self.assertEqual(self._extraction_count(module), 4)
            self.assertTrue(module.operation.verify())

    def test_input_extraction_is_local_to_sibling_blocks(self) -> None:
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
                predicate = constant(True)
                branch = scf.IfOp(native(predicate), [], has_else=True)
                for child in (branch.then_block, branch.else_block):
                    with ir.InsertionPoint(child):
                        values = table.extract(("values", 0))
                        with BindScope([unit], values) as scope:
                            scope.yield_(pure([]))

                        scf.yield_([])

                func.return_([])

            self.assertEqual(self._extraction_count(module), 2)
            self.assertTrue(module.operation.verify())

    def test_container_array_operations(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            with ir.InsertionPoint(importer.module.body):
                integer = constant(7)
                array_type = tvm_ffi_d.ArrayType.get(integer.type)
                created = tvm_ffi_d.array_create(array_type, [integer])
                self.assertEqual(created.type, array_type)
                size = tvm_ffi_d.array_size(created)
                self.assertEqual(size.type, integer.type)
                element = tvm_ffi_d.array_get(created, native(constant(0)))
                self.assertEqual(element.type, integer.type)
                with BindScope(
                    [tvm_ffi_d.ResultType.get(array_type)], pure([created])
                ) as bound:
                    bound_arguments = bound.arguments
                    [array] = bound_arguments
                    union = tvm_ffi_d.UnionType.get(
                        [integer.type, tvm_ffi_d.BoolType.get()]
                    )
                    for target, expected in (
                        (integer.type, integer.type),
                        (tvm_ffi_d.AnyType.get(), tvm_ffi_d.AnyType.get()),
                        (union, union),
                    ):
                        with self.subTest(target=target):
                            [result] = container_get(array, 0, target)
                            self.assertEqual(result.type.payload_type, expected)
                            with BindScope([result.type], [result]) as widened:
                                widened_arguments = widened.arguments
                                [value] = widened_arguments
                                narrowed = tvm_ffi_d.array_create(
                                    tvm_ffi_d.ArrayType.get(expected), [value]
                                )
                                [element] = container_get(narrowed, 0, integer.type)
                                self.assertEqual(
                                    element.type.payload_type, integer.type
                                )
                                widened.yield_(pure([value]))
                    bound.yield_(pure([array]))
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
                integer, boolean = constant(7), constant(True)
                type_ = tvm_ffi_d.TupleType.get([integer.type, boolean.type])
                tuple_ = tvm_ffi_d.tuple_construct(type_, [integer, boolean])
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
                empty = tvm_ffi_d.tuple_construct(empty_type, [])
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
                        with BindScope([none_result], results) as bound:
                            bound_arguments = bound.arguments
                            self.assertEqual(len(bound_arguments), count)
                            self.assertIsInstance(
                                bound_arguments[-1].type, tvm_ffi_d.NoneType
                            )
                            bound.yield_(
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
                    pure([tvm_ffi_d.exception("GuardMatch")])
                    importer.build_arguments(
                        [torch.zeros(2), ("", "1234567", "12345678"), [None, 3]]
                    )
                    func.return_([])
            self.assertTrue(importer.module.operation.verify())
            self.assertNotIn("tvm_ffi.cleanup", f"{importer.module}")
            self.assertNotIn("tvm_ffi.retain", f"{importer.module}")
            check_operation(importer.module.operation)
            parsed = ir.Module.parse(f"{importer.module}")
            check_operation(parsed.operation)

    def test_frontend_scalar_values(self) -> None:
        class Scalars(torch.nn.Module):
            def forward(self, value: torch.Tensor):
                size = value.shape[0]
                return value, (size + 1) ** 2, size > 3, value.stride(0)

        with new_context() as context:
            program = torch.export.export(
                Scalars(),
                (torch.ones(5, 2),),
                dynamic_shapes={"value": {0: torch.export.Dim("size", min=2)}},
            )
            importer = TridentFxImporter(context=context, specialization_id=0)
            importer.import_program(program, func_name="scalars")
            self.assertTrue(importer.module.operation.verify())
            scalar_types = (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
            self.assertTrue(importer.symbols)
            for symbol, value in importer.symbols.items():
                with self.subTest(symbol=symbol):
                    self.assertIsInstance(value.type, scalar_types)
            for node, values in importer.values.items():
                for value in torch.utils._pytree.tree_leaves(values):
                    if isinstance(value, ir.Value):
                        with self.subTest(node=node.name):
                            self.assertNotIsInstance(
                                value.type, (ir.IntegerType, ir.F64Type, ir.IndexType)
                            )
            with ir.Location.unknown(), ir.InsertionPoint(importer.module.body):
                for value in (True, False, -(1 << 63), (1 << 63) - 1, 1.5):
                    with self.subTest(constant=value):
                        result = constant(value)
                        self.assertIsInstance(result.type, scalar_types)
                        self.assertTrue(
                            result.owner.name.startswith("tvm_ffi.constant.")
                        )
                for value in (-(1 << 63) - 1, 1 << 63):
                    with (
                        self.subTest(constant=value),
                        self.assertRaisesRegex(AssertionError, "i64 runtime ABI"),
                    ):
                        constant(value)

    def test_pure_inference_and_typed_boundaries(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            integer = tvm_ffi_d.IntType.get()
            boxed = tvm_ffi_d.IntType.get()
            union = tvm_ffi_d.UnionType.get([boxed, tvm_ffi_d.NoneType.get()])
            with ir.InsertionPoint(module.body):
                value = constant(7)
                inferred = pure([value])
                self.assertEqual(inferred[0].type.payload_type, boxed)
                self.assertEqual(make_result_types(iter([integer])), [inferred[0].type])
                with self.assertRaisesRegex(ValueError, "C ABI"):
                    make_result_types([ir.IntegerType.get_signless(64)])
                with self.assertRaisesRegex(ValueError, "C ABI"):
                    pure([native(value)])
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
                    pure(
                        [value],
                        result_types=make_result_types([tvm_ffi_d.FloatType.get()]),
                    )
                with (
                    self.assertRaisesRegex(RuntimeError, "construction failed"),
                    BindScope(make_result_types([integer]), inferred) as bound,
                ):
                    bound_arguments = bound.arguments
                    bound.yield_(pure(bound_arguments))
                    raise RuntimeError("construction failed")
                self.assertEqual(ir.InsertionPoint.current.block, module.body)
            self.assertTrue(module.operation.verify())

    def test_scope_stack_boundaries(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            types = make_result_types([tvm_ffi_d.IntType.get()])
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("nested", (types, types))
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                with ScopeStack() as outer:
                    payloads = outer.bind(types, block.arguments)
                    conditional = scf.IfOp(native(constant(True)), types, has_else=True)
                    for child in (conditional.then_block, conditional.else_block):
                        with ir.InsertionPoint(child):
                            with ScopeStack() as inner:
                                values = inner.bind(types, pure(payloads))
                                results = inner.yield_(pure(values))
                                self.assertEqual(ir.InsertionPoint.current.block, child)
                                # Draining an empty stack forwards without another terminator.
                                self.assertEqual(inner.yield_(results), results)
                            self.assertEqual(
                                list(child.operations)[-1].operation.name,
                                "tvm_ffi.monad.bind",
                            )
                            scf.yield_(results)
                    self.assertEqual(
                        list(ir.InsertionPoint.current.block.operations)[
                            -1
                        ].operation.name,
                        "scf.if",
                    )
                    results = outer.yield_(conditional.results)
                    self.assertEqual(ir.InsertionPoint.current.block, block)
                func.return_(results)
            self.assertTrue(module.operation.verify())
            self.assertTrue(ir.Module.parse(f"{module}").operation.verify())

            # Construction failure restores the insertion point without adding yields.
            with ir.InsertionPoint(module.body):
                with (
                    self.assertRaisesRegex(RuntimeError, "stop"),
                    ScopeStack() as stack,
                ):
                    scope = stack.push(BindScope(types, pure([constant(1)])))
                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, module.body)
                self.assertEqual(len(scope.block.operations), 0)
                scope.operation.operation.erase()
            self.assertTrue(module.operation.verify())

    def test_unsupported_symbolic_power(self) -> None:
        importer = TridentFxImporter(context=new_context(), specialization_id=0)
        base, exponent = sympy.symbols("base exponent")
        for invalid_exponent in (-1, sympy.Rational(1, 2), exponent):
            with self.subTest(exponent=invalid_exponent):
                expression = sympy.Pow(base, invalid_exponent, evaluate=False)
                with self.assertRaisesRegex(
                    NotImplementedError, "unbound symbolic FX expression"
                ):
                    importer.symbolic(expression)

    def test_scalar_builders(self) -> None:
        cases = (
            (operator.add, True, True, tvm_ffi_d.IntType),
            (operator.add, 2.0, 1, tvm_ffi_d.FloatType),
            (operator.and_, True, False, tvm_ffi_d.BoolType),
        )
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                for operation, lhs, rhs, expected in cases:
                    with self.subTest(operation=operation, lhs=lhs, rhs=rhs):
                        result = BINARY_ARITHMETIC[operation](
                            constant(lhs), constant(rhs)
                        )
                        self.assertIsInstance(result.type, expected)
                with self.assertRaisesRegex(NotImplementedError, "scalar operation"):
                    BINARY_ARITHMETIC[operator.and_](constant(1.0), constant(1))
                for base, exponent, expected in (
                    (True, 0, tvm_ffi_d.IntType),
                    (1.5, 0, tvm_ffi_d.FloatType),
                    (1.5, 3, tvm_ffi_d.FloatType),
                ):
                    with self.subTest(base=base, exponent=exponent):
                        self.assertIsInstance(
                            power(constant(base), exponent).type, expected
                        )
            self.assertTrue(module.operation.verify())
