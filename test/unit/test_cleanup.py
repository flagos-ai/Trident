# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Automatic semantic ownership, exports and block-local cache reuse."""

import unittest

import torch
from trident.core import ir, passmanager, register_all_passes
from trident.core.dialects import arith, func, scf
from trident.core.dialects import tvm_ffi as ffi
from trident.input import InputNodeBuilder, InputTableBuilder
from trident.ir import (
    bind,
    checked_call,
    constant,
    container_create,
    pure,
)
from trident.ownership import is_value_in_scope

from test.support.ir import new_context


class CleanupTest(unittest.TestCase):
    @staticmethod
    def _extraction_count(module: ir.Module) -> int:
        extractions: list[ir.Operation] = []

        def collect(operation: ir.Operation) -> ir.WalkResult:
            if operation.name == "tvm_ffi.array.get":
                extractions.append(operation)
            return ir.WalkResult.ADVANCE

        module.operation.walk(collect)
        return len(extractions)

    def test_bind_get_entry_block(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            payload_types = [ffi.IntType.get(), ffi.NoneType.get()]
            result_types = [ffi.ResultType.get(type_) for type_ in payload_types]
            locations = [ir.Location.name("integer"), ir.Location.name("none")]
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", (result_types, []))
                entry = function.add_entry_block()
            with ir.InsertionPoint(entry):
                operation = ffi.MonadBindOp([], entry.arguments)
                self.assertIsNone(operation.get_entry_block())
                self.assertEqual(len(operation.body.blocks), 0)
                body = operation.add_entry_block(arg_locs=locations)
                self.assertEqual(ir.InsertionPoint.current.block, entry)
                self.assertEqual(
                    [value.type for value in body.arguments], payload_types
                )
                self.assertEqual(
                    [value.location for value in body.arguments], locations
                )
                self.assertEqual(list(operation.arguments), list(body.arguments))
                self.assertEqual(operation.get_entry_block(), body)
                self.assertEqual(operation.add_entry_block(), body)
                self.assertEqual(len(operation.body.blocks), 1)
                self.assertEqual(ir.InsertionPoint.current.block, entry)
                with ir.InsertionPoint(body):
                    ffi.MonadYieldOp([])

                empty = ffi.MonadBindOp([], [])
                self.assertIsNone(empty.get_entry_block())
                with ir.InsertionPoint(empty.add_entry_block()):
                    self.assertEqual(len(empty.arguments), 0)
                    ffi.MonadYieldOp([])

                self.assertEqual(empty.get_entry_block(), empty.body.blocks[0])
                self.assertEqual(empty.add_entry_block(), empty.body.blocks[0])
                self.assertEqual(len(empty.body.blocks), 1)
                func.ReturnOp([])

            self.assertTrue(module.operation.verify())

    def test_bind_arities(self) -> None:
        for input_count, output_count in [(0, 0), (0, 2), (2, 0), (2, 3)]:
            with (
                self.subTest(inputs=input_count, outputs=output_count),
                new_context(),
                ir.Location.unknown(),
            ):
                module = ir.Module.create()
                result_type = ffi.ResultType.get(ffi.IntType.get())
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
                        ffi.MonadYieldOp(values)

                    self.assertEqual(ir.InsertionPoint.current.block, block)
                    self.assertEqual(len(operation.outputs), output_count)
                    func.ReturnOp(operation.outputs)

                self.assertTrue(module.operation.verify())
                register_all_passes()
                pipeline = passmanager.PassManager.parse(
                    "builtin.module(lower-tvm-ffi-monad,decompose-tvm-ffi,lower-tvm-ffi-defer)"
                )
                pipeline.run(module.operation)
                self.assertTrue(module.operation.verify())
                self.assertNotIn("tvm_ffi.monad.bind", f"{module}")

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
                    ffi.MonadYieldOp([])

                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, block)
                func.ReturnOp([])

            self.assertTrue(module.operation.verify())

    def test_cache_lives_until_defining_block_exit(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            array = ffi.ArrayType.get(tensor)
            unit = ffi.ResultType.get(ffi.NoneType.get())
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
                        ffi.MonadYieldOp(pure([]))

                    ffi.MonadYieldOp(reused.outputs)

                # Ending the Python context does not end the IR block lifetime.
                self.assertTrue(is_value_in_scope(first[0]))
                cached = table.extract(("values", 0))
                with bind(cached, result_types=[unit]):
                    ffi.MonadYieldOp(pure([]))

                func.ReturnOp([])

            self.assertEqual(self._extraction_count(module), 1)
            self.assertTrue(module.operation.verify())

    def test_cache_rebuilds_in_sibling_blocks(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            array = ffi.ArrayType.get(tensor)
            unit = ffi.ResultType.get(ffi.NoneType.get())
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
                            ffi.MonadYieldOp(pure([]))

                        scf.YieldOp([])

                func.ReturnOp([])

            self.assertEqual(self._extraction_count(module), 2)
            self.assertTrue(module.operation.verify())

    def test_result_tuple_exports_retain_both_branches(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            result_type = ffi.ResultType.get(ffi.TensorType.get())
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "test", ([ffi.TensorType.get()], [result_type, result_type])
                )
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                inputs = pure([block.arguments[0], block.arguments[0]])
                with bind(inputs, result_types=[result_type, result_type]) as operation:
                    yielded = pure([operation.arguments[0]])
                    ffi.MonadYieldOp([*yielded, *yielded])

                func.ReturnOp(operation.outputs)

            self.assertTrue(module.operation.verify())
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectIncRefOp)
                ],
                [],
            )
            branches: list[ir.Operation] = []

            def collect(operation: ir.Operation) -> ir.WalkResult:
                if operation.name == "scf.if":
                    branches.append(operation)
                return ir.WalkResult.ADVANCE

            module.operation.walk(collect)
            self.assertEqual(branches, [])
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(lower-tvm-ffi-monad,decompose-tvm-ffi,lower-tvm-ffi-defer)"
            )
            pipeline.run(module.operation)
            self.assertTrue(module.operation.verify())
            module.operation.walk(collect)
            self.assertEqual(len(branches), 1)
            success, failure = branches[0].regions
            for region in (success, failure):
                body = region.blocks[0]
                exit_values = list(body.operations[-1].operands)
                retains = [
                    op.object
                    for op in body.operations
                    if isinstance(op, ffi.ObjectIncRefOp)
                ]
                self.assertEqual(retains, exit_values)
                releases = [
                    op.object
                    for op in body.operations
                    if isinstance(op, ffi.ObjectDecRefOp)
                ]
                for value in exit_values:
                    self.assertIn(value, releases)

    def test_frontend_resources_need_no_ownership_attributes(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            result_type = ffi.ResultType.get(tensor)
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("resources", ([tensor], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                constant(torch.ones(1))
                values = checked_call("test.identity", block.arguments, tensor)
                container_create(block.arguments, ffi.ArrayType.get(tensor))
                with bind(values, result_types=[result_type]) as operation:
                    ffi.MonadYieldOp(pure(operation.arguments))
                func.ReturnOp([])
            self.assertTrue(module.operation.verify())
            self.assertNotIn("tvm_ffi.cleanup", f"{module}")
            self.assertNotIn("tvm_ffi.retain", f"{module}")
            register_all_passes()
            passmanager.PassManager.parse(
                "builtin.module(lower-tvm-ffi-monad,decompose-tvm-ffi,lower-tvm-ffi-defer)"
            ).run(module.operation)
            self.assertTrue(module.operation.verify())
            self.assertIn("ObjectDecRef", f"{module}")

    def test_borrowed_alias_and_repeated_exports(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            any_type = ffi.AnyType.get()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("exports", ([], [any_type, any_type]))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                value = ffi.exception("GuardMatch")
                borrowed = ffi.upcast(any_type, value)
                func.ReturnOp([borrowed, borrowed])
            register_all_passes()
            passmanager.PassManager.parse("builtin.module(decompose-tvm-ffi)").run(
                module.operation
            )
            cleanups = [op for op in block.operations if isinstance(op, ffi.DeferOp)]
            self.assertEqual(len(cleanups), 1)
            body = list(cleanups[0].body.blocks[0].operations)
            self.assertIsInstance(body[0], ffi.ObjectDecRefOp)
            self.assertIsInstance(body[-1], ffi.DeferYieldOp)
            self.assertTrue(module.operation.verify())
            passmanager.PassManager.parse("builtin.module(lower-tvm-ffi-defer)").run(
                module.operation
            )
            self.assertNotIn("tvm_ffi.defer", f"{module}")
            operations = list(block.operations)
            exports = list(operations[-1].operands)
            retains = [op for op in operations if isinstance(op, ffi.ObjectIncRefOp)]
            releases = [op for op in operations if isinstance(op, ffi.ObjectDecRefOp)]
            self.assertEqual([op.object for op in retains], exports)
            self.assertEqual(len(releases), 1)
            self.assertIsInstance(releases[0].object.type, ffi.CExceptionType)
            self.assertLess(
                operations.index(retains[-1]), operations.index(releases[0])
            )
            self.assertTrue(module.operation.verify())

    def test_low_level_pure_borrows_repeated_payloads(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("borrow", ([tensor], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                ffi.MonadPureOp(
                    [ffi.ResultType.get(tensor)] * 2, [block.arguments[0]] * 2
                )
                func.ReturnOp([])
            register_all_passes()
            passmanager.PassManager.parse("builtin.module(lower-tvm-ffi-monad)").run(
                module.operation
            )
            self.assertNotIn("ObjectDecRef", f"{module}")
            self.assertEqual(
                len(
                    [
                        op
                        for op in block.operations
                        if isinstance(op, ffi.TupleConstructOp)
                    ]
                ),
                2,
            )
            passmanager.PassManager.parse(
                "builtin.module(decompose-tvm-ffi,lower-tvm-ffi-defer)"
            ).run(module.operation)
            lowered = module.body.operations[0].entry_block
            releases = [
                op.object
                for op in lowered.operations
                if isinstance(op, ffi.ObjectDecRefOp)
            ]
            self.assertEqual(
                sum(isinstance(value.type, ffi.CArrayType) for value in releases), 2
            )
            self.assertNotIn(lowered.arguments[0], releases)
            self.assertTrue(module.operation.verify())
