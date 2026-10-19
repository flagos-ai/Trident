# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Explicit cleanup, explicit exports and block-local cache reuse."""

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
from trident.ownership import CLEANUP_ATTR, RETAIN_ATTR, has_cleanup_owner

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

    def test_cleanup_unit_attribute(self) -> None:
        with new_context(), ir.Location.unknown():
            attribute = ir.UnitAttr.get()
            parsed = ir.Attribute.parse(f"{attribute}")
            self.assertIsInstance(parsed, ir.UnitAttr)
            self.assertEqual(f"{parsed}", "unit")

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
                    terminator = ffi.MonadYieldOp([])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                empty = ffi.MonadBindOp([], [])
                self.assertIsNone(empty.get_entry_block())
                with ir.InsertionPoint(empty.add_entry_block()):
                    self.assertEqual(len(empty.arguments), 0)
                    terminator = ffi.MonadYieldOp([])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                self.assertEqual(empty.get_entry_block(), empty.body.blocks[0])
                self.assertEqual(empty.add_entry_block(), empty.body.blocks[0])
                self.assertEqual(len(empty.body.blocks), 1)
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
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
                        terminator = ffi.MonadYieldOp(values)
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    self.assertEqual(ir.InsertionPoint.current.block, block)
                    self.assertEqual(len(operation.outputs), output_count)
                    terminator = func.ReturnOp(operation.outputs)
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                self.assertTrue(module.operation.verify())
                register_all_passes()
                pipeline = passmanager.PassManager.parse(
                    "builtin.module(materialize-tvm-ffi-cleanup,decompose-tvm-ffi)"
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
                    terminator = ffi.MonadYieldOp([])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    raise RuntimeError("stop")
                self.assertEqual(ir.InsertionPoint.current.block, block)
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertTrue(module.operation.verify())

    def test_borrowed_alias_and_independent_exports(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([], [ffi.AnyType.get()] * 2))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                value = ffi.exception("GuardMatch")
                value.owner.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                borrowed = ffi.upcast(ffi.AnyType.get(), value)
                terminator = func.ReturnOp([borrowed, borrowed])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                self.assertTrue(has_cleanup_owner(value))
                self.assertFalse(has_cleanup_owner(borrowed))
            self.assertTrue(module.operation.verify())
            self.assertNotIn("ObjectIncRef", f"{module}")
            self.assertIn(RETAIN_ATTR, f"{module}")
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            )
            pipeline.run(module.operation)
            materialized = f"{module}"
            pipeline.run(module.operation)
            self.assertEqual(f"{module}", materialized)
            self.assertNotIn(RETAIN_ATTR, materialized)
            references = [
                op for op in block.operations if isinstance(op, ffi.ObjectIncRefOp)
            ]
            self.assertEqual([op.object for op in references], [borrowed, borrowed])
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectDecRefOp)
                ],
                [value],
            )
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
                        terminator = ffi.MonadYieldOp(pure([]))
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    terminator = ffi.MonadYieldOp(reused.outputs)
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                # Ending the Python context does not end the IR block lifetime.
                self.assertTrue(has_cleanup_owner(first[0]))
                cached = table.extract(("values", 0))
                with bind(cached, result_types=[unit]):
                    terminator = ffi.MonadYieldOp(pure([]))
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
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
                            terminator = ffi.MonadYieldOp(pure([]))
                            terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                        terminator = scf.YieldOp([])
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertEqual(self._extraction_count(module), 2)
            self.assertTrue(module.operation.verify())

    def test_created_resources_are_marked_before_binding(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            result_type = ffi.ResultType.get(tensor)
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([tensor], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                literal = constant(torch.ones(1))
                self.assertTrue(has_cleanup_owner(literal))
                call = checked_call("test.identity", block.arguments, tensor)
                container = container_create(block.arguments, ffi.ArrayType.get(tensor))
                for values in (call, container):
                    for value in values:
                        self.assertTrue(has_cleanup_owner(value))
                        self.assertIsInstance(
                            value.owner.attributes[CLEANUP_ATTR], ir.UnitAttr
                        )
                with bind(call, result_types=[result_type]) as operation:
                    self.assertFalse(has_cleanup_owner(operation.arguments[0]))
                    terminator = ffi.MonadYieldOp(pure(operation.arguments))
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                self.assertIsInstance(
                    call[0].owner.attributes[CLEANUP_ATTR], ir.UnitAttr
                )
                self.assertTrue(has_cleanup_owner(operation.outputs[0]))
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertTrue(module.operation.verify())

    def test_explicit_cleanup_and_materialization(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                value = ffi.exception("GuardMatch")
                self.assertFalse(has_cleanup_owner(value))
                value.owner.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                self.assertTrue(has_cleanup_owner(value))
                # Marking an operation twice still releases each result once.
                value.owner.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            )
            pipeline.run(module.operation)
            materialized = f"{module}"
            pipeline.run(module.operation)
            self.assertEqual(f"{module}", materialized)
            with ir.InsertionPoint(block):
                self.assertFalse(has_cleanup_owner(value))
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectDecRefOp)
                ],
                [value],
            )
            self.assertNotIn(CLEANUP_ATTR, materialized)
            self.assertTrue(module.operation.verify())

    def test_scalar_and_empty_cleanup_intent(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                scalar = constant(7)
                self.assertFalse(has_cleanup_owner(scalar))
                scalar.owner.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                self.assertTrue(has_cleanup_owner(scalar))
                [result] = pure([scalar])
                self.assertTrue(has_cleanup_owner(result))
                with bind([], result_types=[]) as empty:
                    terminator = ffi.MonadYieldOp([])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                for operation in (scalar.owner, result.owner, empty):
                    self.assertIsInstance(
                        operation.attributes[CLEANUP_ATTR], ir.UnitAttr
                    )
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertTrue(module.operation.verify())
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            )
            pipeline.run(module.operation)
            materialized = f"{module}"
            self.assertNotIn(CLEANUP_ATTR, materialized)
            self.assertNotIn(RETAIN_ATTR, materialized)
            self.assertNotIn("ObjectIncRef", materialized)
            self.assertNotIn("ObjectDecRef", materialized)
            self.assertTrue(module.operation.verify())
            pipeline.run(module.operation)
            self.assertEqual(f"{module}", materialized)

    def test_marks_all_resource_results_once(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            tensor = ffi.TensorType.get()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "test",
                    ([ffi.TupleType.get([tensor, tensor, ffi.IntType.get()])], []),
                )
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                unpack = ffi.TupleUnconstructOp(
                    [tensor, tensor, ffi.IntType.get()], block.arguments[0]
                )
                unpack.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                self.assertTrue(has_cleanup_owner(unpack.elements[0]))
                self.assertTrue(has_cleanup_owner(unpack.elements[1]))
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            register_all_passes()
            passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            ).run(module.operation)
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectDecRefOp)
                ],
                [unpack.elements[1], unpack.elements[0]],
            )
            self.assertTrue(module.operation.verify())

    def test_container_ownership_is_materialized_after_decomposition(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            container_type = ffi.TupleType.get([])
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([container_type], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                borrowed = block.arguments[0]
                self.assertFalse(has_cleanup_owner(borrowed))
                container = container_create([], container_type)
                with bind(
                    container, result_types=[ffi.ResultType.get(container_type)]
                ) as operation:
                    results = pure([operation.arguments[0], operation.arguments[0]])
                    terminator = ffi.MonadYieldOp(results[:1])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                self.assertTrue(has_cleanup_owner(container[0]))
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertTrue(module.operation.verify())
            self.assertNotIn("ObjectIncRef", f"{module}")
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(decompose-tvm-ffi,materialize-tvm-ffi-cleanup)"
            )
            pipeline.run(module.operation)
            self.assertTrue(module.operation.verify())
            self.assertIsInstance(function.arguments[0].type, ffi.CArrayType)
            materialized = f"{module}"
            self.assertNotIn(RETAIN_ATTR, materialized)
            self.assertNotIn(CLEANUP_ATTR, materialized)
            self.assertNotIn("tvm_ffi.monad.bind", materialized)
            self.assertIn("tvm_ffi.ObjectIncRef", materialized)
            self.assertIn("tvm_ffi.ObjectDecRef", materialized)
            pipeline.run(module.operation)
            self.assertEqual(f"{module}", materialized)

    def test_pure_borrows_each_payload_reference(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", ([], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                value = ffi.exception("GuardMatch")
                value.owner.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                results = pure([value, value, constant(1), constant(None)])
                terminator = func.ReturnOp([])
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertIsInstance(value.owner.attributes[CLEANUP_ATTR], ir.UnitAttr)
            self.assertIsInstance(
                results[0].owner.attributes[CLEANUP_ATTR], ir.UnitAttr
            )
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectIncRefOp)
                ],
                [],
            )
            self.assertIn(RETAIN_ATTR, results[0].owner.attributes)
            self.assertTrue(module.operation.verify())
            register_all_passes()
            passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            ).run(module.operation)
            self.assertTrue(module.operation.verify())
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectIncRefOp)
                ],
                [value, value],
            )
            self.assertNotIn(RETAIN_ATTR, results[0].owner.attributes)
            # Both Results and the source retain their own cleanup responsibility.
            self.assertEqual(
                len([op for op in block.operations if isinstance(op, scf.IfOp)]),
                2,
            )
            self.assertEqual(
                [
                    op.object
                    for op in block.operations
                    if isinstance(op, ffi.ObjectDecRefOp)
                ],
                [value],
            )

            materialized = f"{module}"
            passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            ).run(module.operation)
            self.assertEqual(f"{module}", materialized)

    def test_result_exports_retain_only_on_success(self) -> None:
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
                    terminator = ffi.MonadYieldOp([*yielded, *yielded])
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                terminator = func.ReturnOp(operation.outputs)
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            self.assertTrue(module.operation.verify())
            for value in inputs:
                self.assertIsInstance(value.owner.attributes[CLEANUP_ATTR], ir.UnitAttr)
            self.assertIsInstance(operation.attributes[CLEANUP_ATTR], ir.UnitAttr)
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
            self.assertIn(RETAIN_ATTR, f"{module}")
            register_all_passes()
            pipeline = passmanager.PassManager.parse(
                "builtin.module(materialize-tvm-ffi-cleanup)"
            )
            pipeline.run(module.operation)
            self.assertTrue(module.operation.verify())
            self.assertNotIn(RETAIN_ATTR, f"{module}")
            materialized = f"{module}"
            pipeline.run(module.operation)
            self.assertEqual(f"{module}", materialized)
            module.operation.walk(collect)
            actions = [
                [op.operation.name for op in branch.regions[0].blocks[0].operations]
                for branch in branches
            ]
            self.assertEqual(actions.count(["tvm_ffi.ObjectIncRef", "scf.yield"]), 4)
            self.assertEqual(actions.count(["tvm_ffi.ObjectDecRef", "scf.yield"]), 5)
