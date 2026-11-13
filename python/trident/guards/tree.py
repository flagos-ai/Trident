# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Persistent dispatcher nodes and their native short-circuit IR construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from trident.core import ir
from trident.core.dialects import cf, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import constant, native

from .codes import GuardCode
from .scope import Scope


class Node(ABC):
    """A persistent recipe for one dispatcher node."""

    @abstractmethod
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        """Emit fresh IR and return this node's entry block."""


def _build_children(
    nodes: Sequence[Node],
    scope: Scope,
    fallback_block: ir.Block,
) -> ir.Block:
    for node in reversed(nodes):
        fallback_block = node.build(scope, fallback_block)
    return fallback_block


class ExecutionNode(Node):
    """Forward the original argument pack and return the execution result."""

    def __init__(self, graph_symbol: str) -> None:
        self.graph_symbol = graph_symbol

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        entry = fallback_block.region.blocks.append()
        with ir.InsertionPoint(entry):
            success, payload = tvm_ffi_d.pack_call(
                ir.FlatSymbolRefAttr.get(self.graph_symbol),
                scope.value,
                results=[tvm_ffi_d.BoolType.get(), tvm_ffi_d.AnyType.get()],
            )
            scf.yield_([success, payload])
        return entry


class GuardNode(Node):
    """Branch on one read-only check shared by an ordered subtree."""

    def __init__(self, code: GuardCode) -> None:
        self.children: list[Node] = []
        self.code = code

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


class SCMNode(Node):
    """Bind signature-ordered arguments for a shared candidate subtree."""

    def __init__(self, input_types: Mapping[str, ir.Type]) -> None:
        self.children: list[Node] = []
        self.input_types = dict(input_types)

    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        entry = fallback_block.region.blocks.append()
        with ir.InsertionPoint(entry):
            scope = Scope(
                scope.value,
                {
                    name: Scope(
                        tvm_ffi_d.downcast(
                            type_,
                            tvm_ffi_d.pack_get(scope.value, native(constant(index))),
                        )
                    )
                    for index, (name, type_) in enumerate(self.input_types.items())
                },
            )
        first = _build_children(self.children, scope, fallback_block)
        with ir.InsertionPoint(entry):
            cf.br([], first)
        return entry


class Root:
    """The dispatcher's single persistent description, in specialization order."""

    def __init__(self) -> None:
        self.children: list[SCMNode] = []

    def add(
        self,
        guard_codes: Sequence[GuardCode],
        input_types: Mapping[str, ir.Type],
        graph_symbol: str,
    ) -> None:
        """Extend the trailing common prefix without changing execution order."""
        if self.children and tuple(self.children[-1].input_types.items()) == tuple(
            input_types.items()
        ):
            scm_node = self.children[-1]
        else:
            scm_node = SCMNode(input_types)
            self.children.append(scm_node)
        children = scm_node.children
        for code in guard_codes:
            # Searching earlier siblings could move this execution ahead of
            # intervening candidates. Only the trailing prefix is safe to share.
            if (
                children
                and isinstance(children[-1], GuardNode)
                and children[-1].code.key == code.key
            ):
                guard_node = children[-1]
            else:
                guard_node = GuardNode(code)
                children.append(guard_node)
            children = guard_node.children
        children.append(ExecutionNode(graph_symbol))

    def build(self, scope: Scope) -> Sequence[ir.Value]:
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
        first = _build_children(self.children, scope, miss)
        with ir.InsertionPoint(entry):
            cf.br([], first)
        return machine.results
