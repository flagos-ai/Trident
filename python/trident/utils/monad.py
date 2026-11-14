# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Result builders and lexical scopes for monadic IR construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import ExitStack
from types import TracebackType
from typing import Self, TypeVar

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d


class Scope(ABC):
    """Construct one block with an explicit, operation-specific terminator."""

    def __init__(self, block: ir.Block) -> None:
        self.block: ir.Block = block
        self._insertion_point: ir.InsertionPoint = ir.InsertionPoint(block)

    def __enter__(self) -> Self:
        self._insertion_point.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._insertion_point.__exit__(exc_type, exc_value, traceback)

    @property
    @abstractmethod
    def results(self) -> list[ir.Value]:
        """Complete operation results forwarded to the enclosing scope."""

    @abstractmethod
    def yield_(self, values: Sequence[ir.Value]) -> None:
        """Terminate only this block; operation results belong to its owner."""


class BindScope(Scope):
    """Expose borrowed payloads in one monadic success region."""

    def __init__(
        self, result_types: Sequence[ir.Type], inputs: Sequence[ir.Value]
    ) -> None:
        assert all(isinstance(type_, tvm_ffi_d.ResultType) for type_ in result_types)
        assert all(isinstance(value.type, tvm_ffi_d.ResultType) for value in inputs), (
            "bind inputs must be Results"
        )
        self.operation: tvm_ffi_d.MonadBindOp = tvm_ffi_d.MonadBindOp(
            result_types, inputs
        )
        super().__init__(self.operation.add_entry_block())

    @property
    def arguments(self) -> Sequence[ir.Value]:
        return self.operation.arguments

    @property
    def results(self) -> list[ir.Value]:
        return [*self.operation.outputs]

    def yield_(self, values: Sequence[ir.Value]) -> None:
        with ir.InsertionPoint(self.block):
            tvm_ffi_d.monad_yield(values)


_ScopeT = TypeVar("_ScopeT", bound=Scope)


class ScopeStack(ExitStack):
    """Enter local scopes and explicitly terminate them from innermost outward."""

    def __init__(self) -> None:
        super().__init__()
        self._scopes: list[Scope] = []

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._scopes.clear()

    def bind(
        self, result_types: Sequence[ir.Type], inputs: Sequence[ir.Value]
    ) -> Sequence[ir.Value]:
        return self.push(BindScope(result_types, inputs)).arguments

    def push(self, scope: _ScopeT) -> _ScopeT:
        self.enter_context(scope)
        self._scopes.append(scope)
        return scope

    def yield_(self, values: Sequence[ir.Value]) -> list[ir.Value]:
        """Drain this stack, forwarding complete results from each scope."""
        results: list[ir.Value] = [*values]
        while self._scopes:
            scope = self._scopes.pop()
            scope.yield_(results)
            results = scope.results
        self.close()
        return results


def checked_call(
    name: str, arguments: Sequence[ir.Value], result: ir.Type
) -> list[ir.Value]:
    """Return an asserting invocation wrapped as a successful Result."""
    handle = tvm_ffi_d.FunctionGetGlobalOp(name).result
    call = tvm_ffi_d.FunctionCallOp(result, handle, arguments)
    return [pure(call.result)]


def pure(value: ir.Value, *, result_type: ir.Type | None = None) -> ir.Value:
    """Wrap one borrowed payload in a successful Result."""
    assert isinstance(value, ir.Value), "pure requires one MLIR value"
    if result_type is None:
        result_type = tvm_ffi_d.ResultType.get(value.type)
    assert isinstance(result_type, tvm_ffi_d.ResultType)
    target = result_type.payload_type
    assert value.type == target or isinstance(
        target, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
    ), "payload type must match Result type"
    payload = tvm_ffi_d.upcast(target, value)
    return tvm_ffi_d.MonadPureOp(result_type, payload).output
