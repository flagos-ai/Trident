# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Construct complete Results from normalized guard recipes."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Sequence
from functools import reduce
from types import TracebackType

from trident.core import ir
from trident.core.dialects import scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import (
    BINARY_ARITHMETIC,
    BindScope,
    ScopeStack,
    checked_call,
    constant,
    equal,
    logical_not,
    native,
    negate,
    pure,
    select,
    value_type,
)

from ..scope import Scope
from .compatibility import compatibility_prefix
from .syntax import Expression

_OPERATIONS = {
    "add": operator.add,
    "and": operator.and_,
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
_NUMERIC = (tvm_ffi_d.BoolType, tvm_ffi_d.IntType, tvm_ffi_d.FloatType)


class _LocalScope:
    """Restore the enclosing binding version on every exit."""

    def __init__(self, builder: ExpressionBuilder) -> None:
        self._builder = builder
        self._scope: Scope | None = None

    def __enter__(self) -> None:
        self._scope = self._builder.scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._builder.scope = self._scope


class _OperandScope:
    """Build ordered bind inputs and unwind even if entering fails."""

    def __init__(
        self,
        builder: ExpressionBuilder,
        operands: Sequence[Expression],
        result_type: ir.Type,
    ) -> None:
        self._builder = builder
        self._operands = operands
        self._result_type = result_type
        self._stack = ScopeStack()

    def __enter__(self) -> tuple[ScopeStack, list[ir.Value]]:
        self._stack.__enter__()
        try:
            values: list[ir.Value] = []
            for operand in self._operands:
                result = self._builder._build(operand)
                # Capture after constructing the input, before entering its body.
                self._stack.enter_context(self._builder._local_scope())
                bind_scope = self._stack.push(BindScope([self._result_type], [result]))
                [value] = bind_scope.arguments
                values.append(value)
            return self._stack, values
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self._stack.__exit__(exc_type, exc_value, traceback)


class ExpressionBuilder:
    """An emission-local builder; only payload bindings live in its Scope."""

    def __init__(self, text: str, scope: Scope | None, context: ir.Context) -> None:
        self.text = text
        self.scope = scope
        self.context = context
        self._types: dict[Expression, ir.Type] = {}

    def build(self, expression: Expression, *, predicate: bool = False) -> ir.Value:
        if any(node.kind == "skip" for node in expression.walk()):
            expression, _ = compatibility_prefix(expression)
        try:
            result = self._build(expression)
            if not predicate:
                return result
            type_ = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
            with self._local_scope(), BindScope([type_], [result]) as bind_scope:
                [value] = bind_scope.arguments
                bind_scope.yield_([self._truth(value, self._alternatives(expression))])
            [result] = bind_scope.results
            return result
        except (AssertionError, NotImplementedError) as error:
            raise AssertionError(f"{error}; guard: {self.text!r}") from error

    def _type(self, expression: Expression) -> ir.Type:
        if expression in self._types:
            return self._types[expression]
        kind, operands = expression.kind, expression.operands
        if kind == "constant":
            type_ = value_type(expression.value)
        elif kind == "input":
            assert self.scope is not None, "guard input requires a bound Scope"
            type_ = self.scope.type
            scope: Scope | None = self.scope
            for key in expression.value:
                scope = None if scope is None else scope.get(key)
                if scope is not None:
                    type_ = scope.type
                elif isinstance(type_, tvm_ffi_d.TupleType):
                    type_ = type_.types[key]
                elif isinstance(type_, tvm_ffi_d.ArrayType):
                    type_ = type_.element_type
                else:
                    type_ = tvm_ffi_d.AnyType.get()
        elif kind in ("compare", "eager_compare", "prefix", "skip") or (
            kind == "unary" and expression.value == "not"
        ):
            type_ = tvm_ffi_d.BoolType.get()
        elif kind in ("length", "metadata"):
            type_ = tvm_ffi_d.IntType.get()
        elif kind in ("boolean", "select"):
            choices = operands
            if kind == "select":
                _, *choices = operands
            first, *remaining = (self._type(child) for child in choices)
            type_ = (
                first
                if all(type_ == first for type_ in remaining)
                else tvm_ffi_d.AnyType.get()
            )
        elif kind == "eager_select":
            _, selected, _ = operands
            type_ = self._type(selected)
        elif kind == "eager_boolean":
            first, *_ = operands
            type_ = self._type(first)
        elif kind == "unary":
            [operand] = operands
            type_ = self._type(operand)
            if expression.value == "neg" and isinstance(type_, tvm_ffi_d.BoolType):
                type_ = tvm_ffi_d.IntType.get()
        else:
            types = [self._type(child) for child in operands]
            assert all(isinstance(type_, _NUMERIC) for type_ in types), (
                "arithmetic requires numeric operands"
            )
            if expression.value == "truediv" or any(
                isinstance(type_, tvm_ffi_d.FloatType) for type_ in types
            ):
                type_ = tvm_ffi_d.FloatType.get()
            elif expression.value == "or" and all(
                isinstance(type_, tvm_ffi_d.BoolType) for type_ in types
            ):
                type_ = tvm_ffi_d.BoolType.get()
            else:
                type_ = tvm_ffi_d.IntType.get()
        self._types[expression] = type_
        return type_

    def _alternatives(self, expression: Expression) -> tuple[ir.Type, ...]:
        if expression.kind in ("boolean", "select"):
            operands = expression.operands
            if expression.kind == "select":
                _, *operands = operands
            return tuple(
                dict.fromkeys(
                    type_ for child in operands for type_ in self._alternatives(child)
                )
            )
        return (self._type(expression),)

    def _local_scope(self) -> _LocalScope:
        return _LocalScope(self)

    def _operands(
        self, operands: Sequence[Expression], result_type: ir.Type
    ) -> _OperandScope:
        return _OperandScope(self, operands, result_type)

    def _build(self, expression: Expression) -> ir.Value:
        kind, operands = expression.kind, expression.operands
        result_type = tvm_ffi_d.ResultType.get(self._type(expression))
        if kind == "constant":
            return pure(constant(expression.value))
        if kind == "input":
            assert self.scope is not None
            result, self.scope = self.scope.extract(expression.value)
            return result
        if kind == "skip":
            if expression.value == "base":
                warnings.warn(
                    f"Ignoring unobservable Tensor._base guard: {self.text!r} in {self.context}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return pure(constant(True))
        if kind == "boolean":
            return self._boolean(expression, 0, result_type)
        if kind == "compare":
            first, *_ = operands
            with self._operands([first], result_type) as (stack, values):
                [lhs] = values
                [result] = stack.yield_([self._comparison(expression, 0, lhs)])
                return result
        if kind == "select":
            predicate, *branches = operands
            with self._operands([predicate], result_type) as (stack, values):
                [value] = values
                condition = self._truth(value, self._alternatives(predicate))
                with (
                    self._local_scope(),
                    BindScope([result_type], [condition]) as bind_scope,
                ):
                    [condition_value] = bind_scope.arguments
                    branch = scf.IfOp(
                        native(condition_value), [result_type], has_else=True
                    )
                    for block, child in zip(
                        (branch.then_block, branch.else_block),
                        branches,
                        strict=True,
                    ):
                        with self._local_scope(), ir.InsertionPoint(block):
                            result = self._build(child)
                            if result.type != result_type:
                                with (
                                    self._local_scope(),
                                    BindScope([result_type], [result]) as widening,
                                ):
                                    [value] = widening.arguments
                                    widening.yield_(
                                        [pure(value, result_type=result_type)]
                                    )
                                [result] = widening.results
                            scf.yield_([result])
                    bind_scope.yield_(branch.results)
                [result] = stack.yield_(bind_scope.results)
                return result
        with self._operands(operands, result_type) as (stack, values):
            if kind == "length":
                [value] = values
                result = self._length(value)
            elif kind == "unary" and expression.value == "not":
                [operand], [value] = operands, values
                truth = self._truth(value, self._alternatives(operand))
                with (
                    self._local_scope(),
                    BindScope([result_type], [truth]) as bind_scope,
                ):
                    [value] = bind_scope.arguments
                    bind_scope.yield_([pure(logical_not(value))])
                [result] = bind_scope.results
            else:
                if kind == "prefix":
                    value = constant(True)
                elif kind == "binary":
                    value = BINARY_ARITHMETIC[_OPERATIONS[expression.value]](*values)
                elif kind == "unary":
                    [value] = values
                    if expression.value == "neg":
                        value = negate(value)
                elif kind == "eager_boolean":
                    value = reduce(
                        BINARY_ARITHMETIC[_OPERATIONS[expression.value]], values
                    )
                elif kind == "eager_compare":
                    value = reduce(
                        BINARY_ARITHMETIC[operator.and_], values, constant(True)
                    )
                elif kind == "eager_select":
                    value = select(*values)
                else:
                    assert kind == "metadata"
                    [tensor] = values
                    if not isinstance(tensor.type, tvm_ffi_d.TensorType):
                        tensor = tvm_ffi_d.downcast(tvm_ffi_d.TensorType.get(), tensor)
                    method, index = expression.value
                    if method in ("ndimension", "shape"):
                        value = tvm_ffi_d.to(tvm_ffi_d.tensor_dim(tensor))
                    else:
                        operation = (
                            tvm_ffi_d.tensor_size
                            if method == "size"
                            else tvm_ffi_d.tensor_stride
                        )
                        value = tvm_ffi_d.to(operation(tensor, native(constant(index))))
                result = pure(value)
            [result] = stack.yield_([result])
            return result

    def _boolean(
        self, expression: Expression, index: int, result_type: ir.Type
    ) -> ir.Value:
        operand = expression.operands[index]
        with self._operands([operand], result_type) as (stack, values):
            [value] = values
            if index == len(expression.operands) - 1:
                [result] = stack.yield_([pure(value, result_type=result_type)])
                return result
            truth = self._truth(value, self._alternatives(operand))
            with self._local_scope(), BindScope([result_type], [truth]) as bind_scope:
                [condition] = bind_scope.arguments
                branch = scf.IfOp(native(condition), [result_type], has_else=True)
                for block, continuing in (
                    (branch.then_block, expression.value == "and"),
                    (branch.else_block, expression.value == "or"),
                ):
                    with self._local_scope(), ir.InsertionPoint(block):
                        result = (
                            self._boolean(expression, index + 1, result_type)
                            if continuing
                            else pure(value, result_type=result_type)
                        )
                        scf.yield_([result])
                bind_scope.yield_(branch.results)
            [result] = stack.yield_(bind_scope.results)
            return result

    def _comparison(
        self, expression: Expression, index: int, lhs: ir.Value
    ) -> ir.Value:
        result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
        with self._operands([expression.operands[index + 1]], result_type) as (
            stack,
            values,
        ):
            [rhs] = values
            operation = expression.value[index]
            if operation == "is" and all(
                isinstance(value.type, tvm_ffi_d.TensorType) for value in (lhs, rhs)
            ):
                warnings.warn(
                    f"Skipping unsupported Tensor identity guard; object identity will not be validated: {self.text!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                predicate = constant(True)
            elif operation in ("eq", "is"):
                predicate = equal(lhs, rhs)
            else:
                predicate = BINARY_ARITHMETIC[_OPERATIONS[operation]](lhs, rhs)
            if index == len(expression.value) - 1:
                [result] = stack.yield_([pure(predicate)])
                return result
            branch = scf.IfOp(native(predicate), [result_type], has_else=True)
            with self._local_scope(), ir.InsertionPoint(branch.then_block):
                scf.yield_([self._comparison(expression, index + 1, rhs)])
            with ir.InsertionPoint(branch.else_block):
                scf.yield_([pure(constant(False))])
            [result] = stack.yield_(branch.results)
            return result

    def _length(self, value: ir.Value) -> ir.Value:
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

    def _truth(self, value: ir.Value, alternatives: Sequence[ir.Type]) -> ir.Value:
        type_ = value.type
        if isinstance(type_, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)):
            choices = list(alternatives)
            first, *remaining = choices
            if not remaining and isinstance(
                first, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
            ):
                # Existing untyped input truth guards describe containers.
                return self._truth(
                    tvm_ffi_d.downcast(
                        tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get()), value
                    ),
                    [],
                )
            # Str has no single runtime tag; use it as the final known alternative.
            choices.sort(key=lambda type_: isinstance(type_, tvm_ffi_d.StrType))
            first, *remaining = choices
            if not remaining:
                return self._truth(tvm_ffi_d.downcast(first, value), [])
            result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
            branch = scf.IfOp(tvm_ffi_d.isa(value, first), [result_type], has_else=True)
            with self._local_scope(), ir.InsertionPoint(branch.then_block):
                scf.yield_([self._truth(tvm_ffi_d.downcast(first, value), [])])
            with self._local_scope(), ir.InsertionPoint(branch.else_block):
                scf.yield_([self._truth(value, remaining)])
            [result] = branch.results
            return result
        if isinstance(type_, _NUMERIC):
            return pure(logical_not(logical_not(value)))
        if isinstance(type_, tvm_ffi_d.NoneType):
            return pure(constant(False))
        if isinstance(type_, tvm_ffi_d.StrType):
            return pure(logical_not(equal(value, constant(""))))
        if isinstance(type_, (tvm_ffi_d.ArrayType, tvm_ffi_d.TupleType)):
            result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
            with (
                self._local_scope(),
                BindScope([result_type], [self._length(value)]) as bind_scope,
            ):
                [length] = bind_scope.arguments
                bind_scope.yield_([pure(logical_not(logical_not(length)))])
            [result] = bind_scope.results
            return result
        assert isinstance(type_, (tvm_ffi_d.DTypeType, tvm_ffi_d.DeviceType)), (
            "unsupported guard truth operand"
        )
        return pure(constant(True))
