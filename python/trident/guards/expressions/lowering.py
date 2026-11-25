# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Emission-local bindings and shared Result construction for guard nodes."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from trident.core import ir
from trident.core.dialects import tvm_ffi
from trident.utils.monad import BindScope, ScopeStack, checked_call, pure
from trident.utils.values import constant

from ..scope import Scope

if TYPE_CHECKING:
    from .nodes import Expression


class ExpressionContext:
    """Shared mechanics for one emission; expression behavior lives in nodes."""

    def __init__(self, text: str, scope: Scope | None, context: ir.Context) -> None:
        self.text: Final[str] = text
        self.scope: Scope | None = scope
        self.context: Final[ir.Context] = context

    @contextmanager
    def local_scope(self) -> Generator[None, None, None]:
        scope: Scope | None = self.scope
        try:
            yield
        finally:
            self.scope = scope

    @contextmanager
    def operands(
        self, operands: Sequence[Expression], result_type: ir.Type
    ) -> Generator[tuple[ScopeStack, list[ir.Value]], None, None]:
        with ScopeStack() as stack:
            values: list[ir.Value] = []
            for operand in operands:
                result = operand._build(self)
                # Save the version after constructing the input, before its body.
                stack.enter_context(self.local_scope())
                values.extend(stack.bind([result_type], [result]))
            yield stack, values

    def length(self, value: ir.Value) -> ir.Value:
        if isinstance(value.type, (tvm_ffi.api.AnyType, tvm_ffi.api.UnionType)):
            value = tvm_ffi.api.downcast(
                tvm_ffi.api.ArrayType.get(tvm_ffi.api.AnyType.get()), value
            )
        if isinstance(value.type, tvm_ffi.api.TupleType):
            # Input tuple arity cannot be inferred from its static schema.
            [result] = checked_call("ffi.ArraySize", [value], tvm_ffi.abi.IntType.get())
            return result
        assert isinstance(value.type, tvm_ffi.api.ArrayType), "len requires a container"
        return pure(tvm_ffi.api.array_size(value))

    def truth(self, value: ir.Value, *, invert: bool = False) -> ir.Value:
        """Convert a scalar or runtime container length to a Bool Result."""
        if isinstance(value.type, tvm_ffi.abi.BoolType):
            return pure(tvm_ffi.arith.not_(value) if invert else value)
        if isinstance(value.type, (tvm_ffi.abi.FloatType, tvm_ffi.abi.IntType)):
            # Unordered not-equal preserves NaN truthiness and both zero signs.
            return pure(
                tvm_ffi.arith.not_(value)
                if invert
                else tvm_ffi.arith.ne(value, constant(0))
            )
        result_type: tvm_ffi.monad.ResultType = tvm_ffi.monad.ResultType.get(
            tvm_ffi.abi.BoolType.get()
        )
        with (
            self.local_scope(),
            BindScope([result_type], [self.length(value)]) as bind_scope,
        ):
            [length] = bind_scope.arguments
            bind_scope.yield_([self.truth(length, invert=invert)])
        [result] = bind_scope.results
        return result
