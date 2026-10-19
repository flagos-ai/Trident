# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Result construction contracts and symbolic expression diagnostics."""

import unittest

import sympy
import torch
from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.importer import TridentFxImporter
from trident.ir import (
    bind,
    box,
    constant,
    container_get,
    ffi_type,
    make_result_types,
    native_constant,
    pure,
)

from test.support.ir import new_context


class ImporterTest(unittest.TestCase):
    def test_abi_type_spellings(self) -> None:
        context = new_context()
        with context:
            for name, type_class in (
                ("any", tvm_ffi_d.AnyType),
                ("bool", tvm_ffi_d.BoolType),
                ("c.array", tvm_ffi_d.CArrayType),
                ("dtype", tvm_ffi_d.DTypeType),
                ("device", tvm_ffi_d.DeviceType),
                ("float", tvm_ffi_d.FloatType),
                ("function", tvm_ffi_d.FunctionType),
                ("int", tvm_ffi_d.IntType),
                ("none", tvm_ffi_d.NoneType),
                ("object", tvm_ffi_d.ObjectType),
                ("raw_str", tvm_ffi_d.RawStrType),
                ("small_str", tvm_ffi_d.SmallStrType),
                ("str", tvm_ffi_d.StrType),
                ("tensor", tvm_ffi_d.TensorType),
            ):
                with self.subTest(name=name):
                    type_ = type_class.get()
                    self.assertEqual(f"{type_}", f"!tvm_ffi.{name}")
                    parsed = ffi_type(name)
                    self.assertIsInstance(parsed, type_class)
                    self.assertEqual(parsed, type_)
            exception = tvm_ffi_d.ExceptionType.get()
            self.assertEqual(f"{exception}", "!tvm_ffi.exception")
            self.assertIsInstance(ffi_type("exception"), tvm_ffi_d.ExceptionType)
            union = tvm_ffi_d.UnionType.get([tvm_ffi_d.IntType.get(), exception])
            self.assertEqual(
                f"{union}", "!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>"
            )
            parsed_union = ir.Type.parse(f"{union}")
            self.assertIsInstance(parsed_union, tvm_ffi_d.UnionType)
            self.assertEqual(parsed_union.types, union.types)
            for member in (
                ir.IntegerType.get_signless(64),
                tvm_ffi_d.ObjectType.get(),
                tvm_ffi_d.ResultType.get(tvm_ffi_d.IntType.get()),
            ):
                with (
                    self.subTest(member=f"{member}"),
                    self.assertRaisesRegex(
                        ValueError, "union members must be TVM FFI C ABI types"
                    ),
                ):
                    tvm_ffi_d.UnionType.get([tvm_ffi_d.IntType.get(), member])
            result = tvm_ffi_d.ResultType.get(union)
            self.assertEqual(f"{result}", f"!tvm_ffi.monad.result<{union}>")
            parsed_result = ir.Type.parse(f"{result}")
            self.assertIsInstance(parsed_result, tvm_ffi_d.ResultType)
            self.assertEqual(parsed_result.payload_type, union)

    def test_bind_result_contract(self) -> None:
        context = new_context()
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            module = importer.module
            integer = ir.IntegerType.get_signless(64)
            payload = tvm_ffi_d.ResultType.get(integer)
            unit = tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
            self.assertIsInstance(ir.Type.parse(f"{payload}"), tvm_ffi_d.ResultType)
            self.assertEqual(payload.payload_type, integer)
            self.assertIsInstance(unit.payload_type, tvm_ffi_d.NoneType)
            with self.assertRaises(TypeError):
                tvm_ffi_d.ResultType.get()
            with self.assertRaises(TypeError):
                tvm_ffi_d.ResultType.get(None)
            with self.assertRaisesRegex(ValueError, "nested Result"):
                tvm_ffi_d.ResultType.get(payload)
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
                            self.assertEqual(result.type.payload_type, target)
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
                (["a", "b"], "!tvm_ffi.array<!tvm_ffi.raw_str>"),
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

    def test_container_types(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            integer = tvm_ffi_d.IntType.get()
            array = tvm_ffi_d.ArrayType.get(integer)
            tuple_ = tvm_ffi_d.TupleType.get([array, integer, integer])
            for value, cls in (
                (array, tvm_ffi_d.ArrayType),
                (tuple_, tvm_ffi_d.TupleType),
            ):
                self.assertIsInstance(ir.Type.parse(f"{value}"), cls)
                self.assertEqual(ir.Type.parse(f"{value}"), value)
            self.assertEqual(array.element_type, integer)
            self.assertEqual(tuple_.types, (array, integer, integer))
            abi_members = (
                tvm_ffi_d.AnyType.get(),
                array,
                tvm_ffi_d.ExceptionType.get(),
                tvm_ffi_d.FunctionType.get(),
                tvm_ffi_d.NoneType.get(),
                tuple_,
                tvm_ffi_d.UnionType.get([integer, tvm_ffi_d.NoneType.get()]),
            )
            for element in abi_members:
                with self.subTest(array_element=element):
                    abi_array = tvm_ffi_d.ArrayType.get(element)
                    self.assertEqual(abi_array.element_type, element)
                    self.assertEqual(ir.Type.parse(f"{abi_array}"), abi_array)
            abi_tuple = tvm_ffi_d.TupleType.get(abi_members)
            self.assertEqual(abi_tuple.types, abi_members)
            self.assertEqual(ir.Type.parse(f"{abi_tuple}"), abi_tuple)
            empty = tvm_ffi_d.TupleType.get([])
            self.assertEqual(empty.types, ())
            self.assertEqual(ir.Type.parse(f"{empty}"), empty)
            for type_ in (
                ir.IntegerType.get_signless(64),
                ir.RankedTensorType.get([2], ir.IntegerType.get_signless(64)),
                tvm_ffi_d.ObjectType.get(),
                tvm_ffi_d.ResultType.get(integer),
            ):
                with self.assertRaisesRegex(
                    ValueError, "array element must be a TVM FFI C ABI type"
                ):
                    tvm_ffi_d.ArrayType.get(type_)
                with self.assertRaisesRegex(
                    ValueError, "tuple elements must be TVM FFI C ABI types"
                ):
                    tvm_ffi_d.TupleType.get([type_])
            other = new_context()
            with other:
                foreign = tvm_ffi_d.IntType.get()
            with self.assertRaises(ValueError):
                tvm_ffi_d.ArrayType.get(foreign)
            with self.assertRaises(ValueError):
                tvm_ffi_d.TupleType.get([foreign])

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

    def test_monad_unpack(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                for value in (
                    native_constant(True),
                    native_constant(7),
                    native_constant(2.5),
                    constant(None),
                ):
                    with self.subTest(type=value.type):
                        [result] = pure([value])
                        unpacked = tvm_ffi_d.MonadUnpackOp(result)
                        self.assertEqual(
                            unpacked.success.type, ir.IntegerType.get_signless(1)
                        )
                        expected = (
                            tvm_ffi_d.AnyType.get()
                            if isinstance(value.type, tvm_ffi_d.NoneType)
                            else value.type
                        )
                        self.assertEqual(unpacked.value.type, expected)
                self.assertTrue(module.operation.verify())

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
                self.assertEqual(inferred[0].type.payload_type, integer)
                self.assertEqual(make_result_types(iter([integer])), [inferred[0].type])
                empty = pure([])
                self.assertIsInstance(empty[0].type.payload_type, tvm_ffi_d.NoneType)
                for target in (boxed, tvm_ffi_d.AnyType.get(), union):
                    with self.subTest(target=target):
                        result_types = make_result_types([target])
                        [result] = pure([value], result_types=result_types)
                        operation = result.owner.opview
                        self.assertEqual(operation.values[0].type, target)
                        if target != boxed:
                            widened = operation.values[0]
                            self.assertIsInstance(
                                widened.owner.opview, tvm_ffi_d.UpcastOp
                            )
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
