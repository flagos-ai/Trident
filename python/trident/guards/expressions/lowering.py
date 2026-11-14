# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Emission-local bindings and shared Result construction for guard nodes."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.utils.monad import BindScope, ScopeStack, checked_call, pure
from trident.utils.values import logical_not

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
    def diagnostics(self) -> Generator[None, None, None]:
        try:
            yield
        except (AssertionError, NotImplementedError) as error:
            raise AssertionError(f"{error}; guard: {self.text!r}") from error

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
        if isinstance(value.type, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)):
            value = tvm_ffi_d.downcast(
                tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get()), value
            )
        if isinstance(value.type, tvm_ffi_d.TupleType):
            # Input tuple arity cannot be inferred from its static schema.
            [result] = checked_call("ffi.ArraySize", [value], tvm_ffi_d.IntType.get())
            return result
        assert isinstance(value.type, tvm_ffi_d.ArrayType), "len requires a container"
        return pure(tvm_ffi_d.array_size(value))

    def truth(self, value: ir.Value) -> ir.Value:
        if isinstance(value.type, tvm_ffi_d.BoolType):
            return pure(value)
        if isinstance(
            value.type, (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
        ):
            return pure(logical_not(logical_not(value)))
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            tvm_ffi_d.BoolType.get()
        )
        with (
            self.local_scope(),
            BindScope([result_type], [self.length(value)]) as bind_scope,
        ):
            [length] = bind_scope.arguments
            bind_scope.yield_([pure(logical_not(logical_not(length)))])
        [result] = bind_scope.results
        return result
