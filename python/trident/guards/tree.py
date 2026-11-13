# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Persistent dispatcher nodes and their native short-circuit IR construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from trident.core import ir
from trident.core.dialects import arith, cf, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import constant, native

from .codes import GuardCode
from .scope import Scope


class Node(ABC):
    """A persistent recipe for one dispatcher node."""

    @abstractmethod
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        """Emit fresh IR and return this node's entry block."""


class ExecutionNode(Node):
    """Adapt one graph's arguments and return its terminal result."""

    def __init__(
        self,
        index: int,
        graph_input_paths: Sequence[tuple[int | str, ...]],
        graph_symbol: str,
        graph_input_types: Sequence[ir.Type],
    ) -> None:
        self.graph_input_types = tuple(graph_input_types)
        self.index = index
        self.graph_input_paths = tuple(graph_input_paths)
        self.graph_symbol = graph_symbol

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        entry = fallback_block.region.blocks.append()
        with ir.InsertionPoint(entry):
            result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.AnyType.get())
            arguments: list[ir.Value] = []
            for input_path, type_ in zip(
                self.graph_input_paths, self.graph_input_types, strict=True
            ):
                value, scope = scope.read(input_path)
                if value.type != type_:
                    value = tvm_ffi_d.downcast(type_, value)
                arguments.append(value)
            result = tvm_ffi_d.monad_call(
                result_type, ir.FlatSymbolRefAttr.get(self.graph_symbol), arguments
            )
            success, payload = tvm_ffi_d.monad_unpack(result)
            scf.yield_([tvm_ffi_d.to(success), payload])
        return entry


class GuardNode(Node):
    """Branch on one complete check, propagating failure before its predicate."""

    def __init__(self, code: GuardCode) -> None:
        self.code = code
        self.children: list[Node] = []

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        blocks = fallback_block.region.blocks
        entry, matched, failed = blocks.append(), blocks.append(), blocks.append()
        with ir.InsertionPoint(entry):
            [result] = self.code.build_result(scope, ir.Context.current)
            success, payload = tvm_ffi_d.monad_unpack(result)
            cf.cond_br(success, [], [], matched, failed)
        with ir.InsertionPoint(failed):
            none = tvm_ffi_d.upcast(tvm_ffi_d.AnyType.get(), constant(None))
            scf.yield_([tvm_ffi_d.to(success), none])
        passed = _build_children(self.children, scope, fallback_block)
        with ir.InsertionPoint(matched):
            predicate = tvm_ffi_d.downcast(tvm_ffi_d.BoolType.get(), payload)
            cf.cond_br(native(predicate), [], [], passed, fallback_block)
        return entry


class InputNode(Node):
    """Own static input types and validate signature-ordered root arguments."""

    def __init__(self, input_types: Mapping[str, ir.Type]) -> None:
        self.input_types = dict(input_types)
        self.children: list[Node] = []

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        blocks = fallback_block.region.blocks
        entry, slots, typed = blocks.append(), blocks.append(), blocks.append()
        types = tuple(self.input_types.values())
        with ir.InsertionPoint(entry):
            count = tvm_ffi_d.pack_size(scope.value)
            cf.cond_br(
                arith.cmpi(arith.CmpIPredicate.eq, count, native(constant(len(types)))),
                [],
                [],
                slots,
                fallback_block,
            )
        with ir.InsertionPoint(slots):
            tags = native(constant(True))
            values: list[ir.Value] = []
            for index, type_ in enumerate(types):
                slot = tvm_ffi_d.pack_get(scope.value, native(constant(index)))
                values.append(slot)
                # Match lower-to-pack's erased Str/Union and runtime Array tags.
                if isinstance(
                    type_, (tvm_ffi_d.AnyType, tvm_ffi_d.StrType, tvm_ffi_d.UnionType)
                ):
                    continue
                if isinstance(type_, (tvm_ffi_d.ArrayType, tvm_ffi_d.TupleType)):
                    type_ = tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get())
                tags = arith.andi(tags, tvm_ffi_d.isa(slot, type_))
            cf.cond_br(tags, [], [], typed, fallback_block)
        with ir.InsertionPoint(typed):
            typed_values = [
                tvm_ffi_d.downcast(type_, value)
                for type_, value in zip(types, values, strict=True)
            ]
        scope = Scope(
            scope.value,
            {
                name: Scope(value)
                for name, value in zip(self.input_types, typed_values, strict=True)
            },
        )
        first = _build_children(self.children, scope, fallback_block)
        with ir.InsertionPoint(typed):
            cf.br([], first)
        return entry


class Root(Node):
    """The dispatcher's single persistent description, in specialization order."""

    def __init__(self) -> None:
        self.children: list[InputNode] = []
        self.size = 0

    def add(
        self,
        guard_codes: Sequence[GuardCode],
        input_types: Mapping[str, ir.Type],
        graph_input_paths: Sequence[tuple[int | str, ...]],
        graph_symbol: str,
        *,
        graph_input_types: Sequence[ir.Type] | None = None,
    ) -> None:
        """Append an independent candidate in specialization order."""
        input_node = InputNode(input_types)
        self.children.append(input_node)
        children = input_node.children
        for code in guard_codes:
            guard_node = GuardNode(code)
            children.append(guard_node)
            children = guard_node.children
        if graph_input_types is None:
            inferred_types: list[ir.Type] = []
            for name, *keys in graph_input_paths:
                type_ = input_types[name]
                for key in keys:
                    type_ = (
                        type_.types[key]
                        if isinstance(type_, tvm_ffi_d.TupleType)
                        else type_.element_type
                    )
                inferred_types.append(type_)
            graph_input_types = inferred_types
        children.append(
            ExecutionNode(self.size, graph_input_paths, graph_symbol, graph_input_types)
        )
        self.size += 1

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        return _build_children(self.children, scope, fallback_block)


def build_dispatcher(root: Root, scope: Scope) -> Sequence[ir.Value]:
    """Build a fresh dispatch region without retaining SSA in the tree."""
    machine = scf.ExecuteRegionOp(
        [tvm_ffi_d.BoolType.get(), tvm_ffi_d.AnyType.get()], no_inline=True
    )
    entry, miss = machine.region.blocks.append(), machine.region.blocks.append()
    with ir.InsertionPoint(miss):
        rejection = tvm_ffi_d.upcast(
            tvm_ffi_d.AnyType.get(), tvm_ffi_d.exception("GuardMiss")
        )
        scf.yield_([constant(True), rejection])
    first = root.build(scope, miss)
    with ir.InsertionPoint(entry):
        cf.br([], first)
    return machine.results


def _build_children(
    nodes: Sequence[Node],
    scope: Scope,
    fallback_block: ir.Block,
) -> ir.Block:
    for node in reversed(nodes):
        fallback_block = node.build(scope, fallback_block)
    return fallback_block
