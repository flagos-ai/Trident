# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Persistent dispatcher nodes and their native short-circuit IR construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import override

from trident.core import ir
from trident.core.dialects.tvm_ffi import monad as tvm_ffi_monad
from trident.core.dialects import builtin, cf, scf
from trident.core.dialects.tvm_ffi import abi as tvm_ffi_abi
from trident.core.dialects.tvm_ffi import api as tvm_ffi_api
from trident.utils.values import constant, native

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
    """Execute one candidate and fall through only on a specialization reject."""

    def __init__(self, graph_symbol: str) -> None:
        self.graph_symbol: Final[str] = graph_symbol

    @override
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        blocks = fallback_block.region.blocks
        bool_type = tvm_ffi_abi.BoolType.get()
        any_type = tvm_ffi_api.AnyType.get()
        entry = blocks.append()
        classify = blocks.append(bool_type, any_type)
        exception = blocks.append(bool_type, any_type)
        rejected = blocks.append(any_type)
        terminal = blocks.append(bool_type, any_type)
        with ir.InsertionPoint(entry):
            success, payload = tvm_ffi_api.pack_call(
                ir.FlatSymbolRefAttr.get(self.graph_symbol),
                scope.value,
                results=[bool_type, any_type],
            )
            # CFG arguments borrow their payload. Keep one reference alive
            # after pack_call's block-local result cleanup runs. Reference counting
            # uses ABI values; API-to-ABI lowering removes these cast bridges.
            borrowed = builtin.UnrealizedConversionCastOp(
                [tvm_ffi_abi.AnyType.get()], [payload]
            ).results[0]
            tvm_ffi_abi.ObjectIncRefOp(borrowed)
            cf.cond_br(
                native(success),
                [success, payload],
                [success, payload],
                classify,
                terminal,
            )
        with ir.InsertionPoint(classify):
            classify_success, classify_payload = classify.arguments
            is_exception = tvm_ffi_api.isa(
                classify_payload,
                tvm_ffi_api.ExceptionType.get(),
            )
            cf.cond_br(
                is_exception,
                [classify_success, classify_payload],
                [classify_success, classify_payload],
                exception,
                terminal,
            )
        with ir.InsertionPoint(exception):
            exception_success, exception_payload = exception.arguments
            handle = tvm_ffi_api.FunctionGetGlobalOp(
                "trident.ffi.IsCandidateReject"
            ).result
            is_rejection = tvm_ffi_api.FunctionCallOp(
                tvm_ffi_abi.BoolType.get(),
                handle,
                [exception_success, exception_payload],
            ).result
            cf.cond_br(
                native(is_rejection),
                [exception_payload],
                [exception_success, exception_payload],
                rejected,
                terminal,
            )
        with ir.InsertionPoint(rejected):
            [rejected_payload] = rejected.arguments
            borrowed = builtin.UnrealizedConversionCastOp(
                [tvm_ffi_abi.AnyType.get()], [rejected_payload]
            ).results[0]
            tvm_ffi_abi.ObjectDecRefOp(borrowed)
            cf.br([], fallback_block)
        with ir.InsertionPoint(terminal):
            terminal_success, terminal_payload = terminal.arguments
            cleanup = tvm_ffi_abi.DeferOp()
            with ir.InsertionPoint(cleanup.body.blocks.append()):
                borrowed = builtin.UnrealizedConversionCastOp(
                    [tvm_ffi_abi.AnyType.get()], [terminal_payload]
                ).results[0]
                tvm_ffi_abi.ObjectDecRefOp(borrowed)
                tvm_ffi_abi.DeferYieldOp()
            scf.yield_([terminal_success, terminal_payload])
        return entry


class CandidateGroup(Node):
    """An ordered set of executions sharing one complete Dynamo guard path."""

    def __init__(self, graph_symbol: str) -> None:
        self.children: list[ExecutionNode] = [ExecutionNode(graph_symbol)]

    def append(self, graph_symbol: str) -> None:
        if all(child.graph_symbol != graph_symbol for child in self.children):
            self.children.append(ExecutionNode(graph_symbol))

    @override
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        return _build_children(self.children, scope, fallback_block)


class GuardNode(Node):
    """Branch on one read-only check shared by an ordered subtree."""

    def __init__(self, code: GuardCode) -> None:
        self.children: list[Node] = []
        self.code: Final[GuardCode] = code

    @override
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        blocks = fallback_block.region.blocks
        entry, matched, failed = blocks.append(), blocks.append(), blocks.append()
        with ir.InsertionPoint(entry):
            [result] = self.code.build_result(scope, ir.Context.current)
            success, payload = tvm_ffi_monad.unpack(result)
            cf.cond_br(success, [], [], matched, failed)
        with ir.InsertionPoint(failed):
            none = tvm_ffi_api.upcast(tvm_ffi_api.AnyType.get(), constant(None))
            scf.yield_([tvm_ffi_abi.to(success), none])
        passed: ir.Block = _build_children(self.children, scope, fallback_block)
        with ir.InsertionPoint(matched):
            predicate = tvm_ffi_api.downcast(tvm_ffi_abi.BoolType.get(), payload)
            cf.cond_br(native(predicate), [], [], passed, fallback_block)
        return entry


class SCMNode(Node):
    """Bind signature-ordered arguments for a shared candidate subtree."""

    def __init__(self, input_types: Mapping[str, ir.Type]) -> None:
        self.children: list[Node] = []
        self.input_types: Final[dict[str, ir.Type]] = dict(input_types)

    @override
    def build(self, scope: Scope, fallback_block: ir.Block) -> ir.Block:
        entry: ir.Block = fallback_block.region.blocks.append()
        with ir.InsertionPoint(entry):
            scope = Scope(
                scope.value,
                {
                    name: Scope(
                        tvm_ffi_api.downcast(
                            type_,
                            tvm_ffi_api.pack_get(scope.value, native(constant(index))),
                        )
                    )
                    for index, (name, type_) in enumerate(self.input_types.items())
                },
            )
        first: ir.Block = _build_children(self.children, scope, fallback_block)
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
        scm_node: SCMNode
        guard_node: GuardNode
        if self.children and tuple(self.children[-1].input_types.items()) == tuple(
            input_types.items()
        ):
            scm_node = self.children[-1]
        else:
            scm_node = SCMNode(input_types)
            self.children.append(scm_node)
        children: list[Node] = scm_node.children
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
        if children and isinstance(children[-1], CandidateGroup):
            children[-1].append(graph_symbol)
        else:
            children.append(CandidateGroup(graph_symbol))

    def build(self, scope: Scope) -> Sequence[ir.Value]:
        """Build a fresh dispatch region without retaining SSA in the tree."""
        machine = scf.ExecuteRegionOp(
            [tvm_ffi_abi.BoolType.get(), tvm_ffi_api.AnyType.get()], no_inline=True
        )
        entry, miss = machine.region.blocks.append(), machine.region.blocks.append()
        with ir.InsertionPoint(miss):
            rejection = tvm_ffi_api.upcast(
                tvm_ffi_api.AnyType.get(), tvm_ffi_api.exception("GuardMiss")
            )
            scf.yield_([constant(True), rejection])
        first: ir.Block = _build_children(self.children, scope, miss)
        with ir.InsertionPoint(entry):
            cf.br([], first)
        return machine.results
