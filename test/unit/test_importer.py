# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Result construction contracts and symbolic expression diagnostics."""

import unittest

import sympy
from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.importer import TridentFxImporter
from trident.ir import bind, constant, make_result_types, native_constant, pure

from test.support.ir import new_context


class ImporterTest(unittest.TestCase):
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
                for inputs in ([], [value]):
                    with self.subTest(inputs=inputs):
                        count = len(module.body.operations)
                        with (
                            self.assertRaisesRegex(AssertionError, "bind inputs"),
                            bind(inputs, result_types=[payload]),
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

    def test_empty_arguments_and_explicit_none(self) -> None:
        def collect(operation: ir.Operation) -> list[tvm_ffi_d.FunctionCallOp]:
            calls: list[tvm_ffi_d.FunctionCallOp] = []
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        if isinstance(child, tvm_ffi_d.FunctionCallOp):
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
                        self.assertEqual(len(call.arguments), 0)

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
