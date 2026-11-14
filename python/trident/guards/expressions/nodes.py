# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Guard nodes with their own typing and Result construction."""

from __future__ import annotations

import operator
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal

import torch

from trident.core import ir
from trident.core.dialects import scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import (
    BINARY_ARITHMETIC,
    BindScope,
    constant,
    equal,
    logical_not,
    native,
    negate,
    pure,
    value_type,
)

if TYPE_CHECKING:
    from ..scope import Scope
    from .lowering import ExpressionContext

BinaryOperator = Literal[
    "add", "floordiv", "max", "min", "mod", "mul", "or", "sub", "truediv"
]
BooleanOperator = Literal["and", "or"]
ComparisonOperator = Literal["eq", "ge", "gt", "is", "le", "lt", "ne"]
LiteralValue = bool | int | float | str | None | torch.device | torch.dtype
UnaryOperator = Literal["neg", "not", "pos"]

_OPERATIONS: Final[dict[BinaryOperator | ComparisonOperator, Callable[..., Any]]] = {
    "add": operator.add,
    "eq": operator.eq,
    "floordiv": operator.floordiv,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "max": max,
    "min": min,
    "mod": operator.mod,
    "mul": operator.mul,
    "ne": operator.ne,
    "or": operator.or_,
    "sub": operator.sub,
    "truediv": operator.truediv,
}


class UnsupportedExpression(ValueError):
    """Syntax or metadata outside the supported Dynamo guard language."""


class _UnobservableMetadata(Exception):
    """Metadata that can only be consumed by relaxing a comparison."""

    def __init__(self, message: str, *, warning: str | None = None) -> None:
        super().__init__(message)
        self.warning: Final[str | None] = warning


class Expression(ABC):
    """A reusable recipe containing no emission-local types or SSA values."""

    def build(self, context: ExpressionContext) -> ir.Value:
        with context.diagnostics():
            try:
                return self._build(context)
            except _UnobservableMetadata as error:
                raise UnsupportedExpression(f"{error}") from None

    @abstractmethod
    def infer_type(self, context: ExpressionContext) -> ir.Type:
        """Infer the payload type without emitting IR or caching the result."""

    @abstractmethod
    def _build(self, context: ExpressionContext) -> ir.Value:
        """Construct one complete Result in the current lexical scope."""

    def children(self) -> tuple[Expression, ...]:
        return ()


class BaseExpression(Expression):
    def __init__(self, operand: Expression) -> None:
        self.operand: Final[Expression] = operand

    def children(self) -> tuple[Expression, ...]:
        return (self.operand,)

    def _error(self) -> _UnobservableMetadata:
        return _UnobservableMetadata(
            "Tensor._base requires a local comparison",
            warning="Ignoring unobservable Tensor._base guard comparison",
        )

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        raise self._error()

    def _build(self, context: ExpressionContext) -> ir.Value:
        raise self._error()


class BinaryExpression(Expression):
    def __init__(
        self, lhs: Expression, rhs: Expression, operation: BinaryOperator
    ) -> None:
        self.lhs: Final[Expression] = lhs
        self.rhs: Final[Expression] = rhs
        self.operation: Final[BinaryOperator] = operation

    def children(self) -> tuple[Expression, ...]:
        return self.lhs, self.rhs

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        types: list[ir.Type] = [child.infer_type(context) for child in self.children()]
        assert all(
            isinstance(
                type_, (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
            )
            for type_ in types
        ), "arithmetic requires numeric operands"
        if self.operation == "truediv" or any(
            isinstance(type_, tvm_ffi_d.FloatType) for type_ in types
        ):
            return tvm_ffi_d.FloatType.get()
        if self.operation == "or" and all(
            isinstance(type_, tvm_ffi_d.BoolType) for type_ in types
        ):
            return tvm_ffi_d.BoolType.get()
        return tvm_ffi_d.IntType.get()

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands(self.children(), result_type) as (stack, values):
            value = BINARY_ARITHMETIC[_OPERATIONS[self.operation]](*values)
            [result] = stack.yield_([pure(value)])
            return result


class BooleanExpression(Expression):
    def __init__(
        self, operands: tuple[Expression, ...], operation: BooleanOperator
    ) -> None:
        self.operands: Final[tuple[Expression, ...]] = operands
        self.operation: Final[BooleanOperator] = operation

    def children(self) -> tuple[Expression, ...]:
        return self.operands

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        assert all(
            isinstance(child.infer_type(context), tvm_ffi_d.BoolType)
            for child in self.operands
        ), "boolean guards require Bool operands"
        return tvm_ffi_d.BoolType.get()

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        predicate, *remaining = self.operands
        if not remaining:
            return predicate._build(context)
        with context.operands([predicate], result_type) as (stack, values):
            [value] = values
            truth = context.truth(value)
            with context.local_scope(), BindScope([result_type], [truth]) as bind_scope:
                [condition] = bind_scope.arguments
                branch: scf.IfOp = scf.IfOp(
                    native(condition), [result_type], has_else=True
                )
                tail = BooleanExpression(tuple(remaining), self.operation)
                selected, other = (
                    (branch.then_block, branch.else_block)
                    if self.operation == "and"
                    else (branch.else_block, branch.then_block)
                )
                with context.local_scope(), ir.InsertionPoint(selected):
                    scf.yield_([tail._build(context)])
                with ir.InsertionPoint(other):
                    scf.yield_([pure(value)])
                bind_scope.yield_(branch.results)
            [result] = stack.yield_(bind_scope.results)
            return result


class ComparisonExpression(Expression):
    def __init__(
        self,
        operands: tuple[Expression, ...],
        operations: tuple[ComparisonOperator, ...],
        *,
        text: str = "",
    ) -> None:
        self.operands: Final[tuple[Expression, ...]] = operands
        self.operations: Final[tuple[ComparisonOperator, ...]] = operations
        self.text: Final[str] = text

    def children(self) -> tuple[Expression, ...]:
        return self.operands

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        return tvm_ffi_d.BoolType.get()

    def _build(self, context: ExpressionContext) -> ir.Value:
        block = ir.InsertionPoint.current.block
        existing: set[ir.Operation | ir.OpView] = set(block.operations)
        scope: Scope | None = context.scope
        try:
            result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
                self.infer_type(context)
            )
            first, *_ = self.operands
            with context.operands([first], result_type) as (stack, values):
                [lhs] = values
                [result] = stack.yield_([self._comparison(context, 0, lhs)])
                return result
        except _UnobservableMetadata as error:
            # Discard this comparison's incomplete regions after scopes unwind.
            context.scope = scope
            for operation in reversed(list(block.operations)):
                if operation not in existing:
                    operation.operation.erase()
            if error.warning is not None:
                warnings.warn(
                    f"{error.warning}: {self.text!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return pure(constant(True))

    def _comparison(
        self, context: ExpressionContext, index: int, lhs: ir.Value
    ) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands([self.operands[index + 1]], result_type) as (
            stack,
            values,
        ):
            [rhs] = values
            operation = self.operations[index]
            if operation == "is" and all(
                isinstance(value.type, tvm_ffi_d.TensorType) for value in (lhs, rhs)
            ):
                warnings.warn(
                    f"Skipping unsupported Tensor identity guard; object identity will not be validated: {context.text!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                predicate = constant(True)
            elif operation in ("eq", "is"):
                predicate = equal(lhs, rhs)
            else:
                predicate = BINARY_ARITHMETIC[_OPERATIONS[operation]](lhs, rhs)
            if index == len(self.operations) - 1:
                [result] = stack.yield_([pure(predicate)])
                return result
            branch: scf.IfOp = scf.IfOp(native(predicate), [result_type], has_else=True)
            with context.local_scope(), ir.InsertionPoint(branch.then_block):
                scf.yield_([self._comparison(context, index + 1, rhs)])
            with ir.InsertionPoint(branch.else_block):
                scf.yield_([pure(constant(False))])
            [result] = stack.yield_(branch.results)
            return result


class ConstantExpression(Expression):
    def __init__(self, value: LiteralValue) -> None:
        self.value: Final[LiteralValue] = value

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        return value_type(self.value)

    def _build(self, context: ExpressionContext) -> ir.Value:
        return pure(constant(self.value))


class InputExpression(Expression):
    def __init__(self, path: list[int | str]) -> None:
        self.path: Final[list[int | str]] = path

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        return context.input_type(self.path)

    def _build(self, context: ExpressionContext) -> ir.Value:
        return context.extract(self.path)


class LengthExpression(Expression):
    def __init__(self, operand: Expression) -> None:
        self.operand: Final[Expression] = operand

    def children(self) -> tuple[Expression, ...]:
        return (self.operand,)

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        return tvm_ffi_d.IntType.get()

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands(self.children(), result_type) as (stack, values):
            [value] = values
            [result] = stack.yield_([context.length(value)])
            return result


class TensorMetadataExpression(Expression):
    """Shared tensor narrowing and Result construction for metadata nodes."""

    def __init__(self, operand: Expression) -> None:
        self.operand: Final[Expression] = operand

    def children(self) -> tuple[Expression, ...]:
        return (self.operand,)

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        return tvm_ffi_d.IntType.get()

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands(self.children(), result_type) as (stack, values):
            [tensor] = values
            if not isinstance(tensor.type, tvm_ffi_d.TensorType):
                tensor = tvm_ffi_d.downcast(tvm_ffi_d.TensorType.get(), tensor)
            [result] = stack.yield_([pure(tvm_ffi_d.to(self._metadata(tensor)))])
            return result

    @abstractmethod
    def _metadata(self, tensor: ir.Value) -> ir.Value:
        """Read the concrete native tensor metadata value."""


class RankExpression(TensorMetadataExpression):
    def _metadata(self, tensor: ir.Value) -> ir.Value:
        return tvm_ffi_d.tensor_dim(tensor)


class SelectExpression(Expression):
    def __init__(
        self, predicate: Expression, selected: Expression, alternative: Expression
    ) -> None:
        self.predicate: Final[Expression] = predicate
        self.selected: Final[Expression] = selected
        self.alternative: Final[Expression] = alternative

    def children(self) -> tuple[Expression, ...]:
        return self.predicate, self.selected, self.alternative

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        type_ = self.selected.infer_type(context)
        assert type_ == self.alternative.infer_type(context), (
            "conditional guard branches must have the same type"
        )
        return type_

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands([self.predicate], result_type) as (stack, values):
            [value] = values
            truth = context.truth(value)
            with context.local_scope(), BindScope([result_type], [truth]) as bind_scope:
                [condition] = bind_scope.arguments
                branch: scf.IfOp = scf.IfOp(
                    native(condition), [result_type], has_else=True
                )
                for block, choice in zip(
                    (branch.then_block, branch.else_block),
                    (self.selected, self.alternative),
                    strict=True,
                ):
                    with context.local_scope(), ir.InsertionPoint(block):
                        scf.yield_([choice._build(context)])
                bind_scope.yield_(branch.results)
            [result] = stack.yield_(bind_scope.results)
            return result


class SizeExpression(TensorMetadataExpression):
    def __init__(self, operand: Expression, index: int) -> None:
        super().__init__(operand)
        self.index: Final[int] = index

    def _metadata(self, tensor: ir.Value) -> ir.Value:
        return tvm_ffi_d.tensor_size(tensor, native(constant(self.index)))


class StorageOffsetExpression(Expression):
    def __init__(self, operand: Expression) -> None:
        self.operand: Final[Expression] = operand

    def children(self) -> tuple[Expression, ...]:
        return (self.operand,)

    def _error(self) -> _UnobservableMetadata:
        return _UnobservableMetadata("storage_offset requires a local comparison")

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        raise self._error()

    def _build(self, context: ExpressionContext) -> ir.Value:
        raise self._error()


class StrideExpression(TensorMetadataExpression):
    def __init__(self, operand: Expression, index: int) -> None:
        super().__init__(operand)
        self.index: Final[int] = index

    def _metadata(self, tensor: ir.Value) -> ir.Value:
        return tvm_ffi_d.tensor_stride(tensor, native(constant(self.index)))


class UnaryExpression(Expression):
    def __init__(self, operand: Expression, operation: UnaryOperator) -> None:
        self.operand: Final[Expression] = operand
        self.operation: Final[UnaryOperator] = operation

    def children(self) -> tuple[Expression, ...]:
        return (self.operand,)

    def infer_type(self, context: ExpressionContext) -> ir.Type:
        if self.operation == "not":
            return tvm_ffi_d.BoolType.get()
        type_ = self.operand.infer_type(context)
        if self.operation == "neg" and isinstance(type_, tvm_ffi_d.BoolType):
            return tvm_ffi_d.IntType.get()
        return type_

    def _build(self, context: ExpressionContext) -> ir.Value:
        result_type: tvm_ffi_d.ResultType = tvm_ffi_d.ResultType.get(
            self.infer_type(context)
        )
        with context.operands(self.children(), result_type) as (stack, values):
            [value] = values
            if self.operation == "not":
                truth = context.truth(value)
                with (
                    context.local_scope(),
                    BindScope([result_type], [truth]) as bind_scope,
                ):
                    [value] = bind_scope.arguments
                    bind_scope.yield_([pure(logical_not(value))])
                [result] = bind_scope.results
            else:
                if self.operation == "neg":
                    value = negate(value)
                result = pure(value)
            [result] = stack.yield_([result])
            return result
