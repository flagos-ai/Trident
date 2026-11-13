# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from itertools import pairwise
from typing import Any, ClassVar

import torch

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import (
    BINARY_ARITHMETIC,
    BindScope,
    ScopeStack,
    checked_call,
    constant,
    equal,
    logical_not,
    make_result_types,
    native,
    negate,
    pure,
    select,
    value_type,
)

from .scope import Scope, input_source


class ASTVisitor:
    """Validate guard syntax without creating IR or delayed builder chains."""

    scalar_ops: ClassVar[
        dict[type[ast.operator | ast.cmpop] | str, Callable[..., Any]]
    ] = {
        ast.Add: operator.add,
        ast.BitOr: operator.or_,
        ast.Div: operator.truediv,
        ast.Eq: operator.eq,
        ast.FloorDiv: operator.floordiv,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Mod: operator.mod,
        ast.Mult: operator.mul,
        ast.NotEq: operator.ne,
        ast.Sub: operator.sub,
        "max": max,
        "min": min,
    }

    @classmethod
    def _supports_call(cls, node: ast.Call) -> bool:
        assert isinstance(node.func, ast.Name)
        if cls.device(node) is not None:
            return True
        if node.keywords:
            return False
        if node.func.id in ("max", "min") and len(node.args) == 2:
            return all(cls.supports(value) for value in node.args)
        if node.func.id == "len" and len(node.args) == 1:
            (source,) = node.args
            if isinstance(source, ast.Attribute) and source.attr == "shape":
                source = source.value
            return input_source(source) is not None
        return False

    @classmethod
    def _supports_comparison(cls, node: ast.Compare) -> bool:
        # Leave invalid chains and other operators to the builder's diagnostics.
        if (
            not any(isinstance(operation, ast.Is) for operation in node.ops)
            or len(node.ops) != 1
        ):
            return True
        (rhs,) = node.comparators
        if isinstance(rhs, ast.Constant) and rhs.value is None:
            return cls.supports(node.left)
        left, right = input_source(node.left), input_source(rhs)
        return (
            left is not None and right is not None and ast.dump(left) != ast.dump(right)
        )

    @classmethod
    def _supports_metadata(cls, node: ast.expr) -> bool:
        source, method, index = cls.metadata(node)
        if source is not None:
            return (method in ("ndimension", "storage_offset") and index is None) or (
                method in ("size", "stride") and index is not None
            )
        call = node.value if isinstance(node, ast.Subscript) else node
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "_base"
            and not call.args
            and not call.keywords
        )

    @staticmethod
    def device(node: ast.Call) -> torch.device | None:
        if not isinstance(node.func, ast.Name) or node.func.id != "device" or node.args:
            return None
        try:
            keywords = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
            }
            return (
                torch.device(**keywords)
                if len(keywords) == len(node.keywords)
                else None
            )
        except (SyntaxError, TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def metadata(node: ast.expr) -> tuple[ast.Subscript | None, str, int | None]:
        index = None
        if isinstance(node, ast.Subscript):
            if (
                not isinstance(node.slice, ast.Constant)
                or not isinstance(node.slice.value, int)
                or isinstance(node.slice.value, bool)
            ):
                return None, "", None
            index = node.slice.value
            node = node.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and not node.args
            and not node.keywords
        ):
            return input_source(node.func.value), node.func.attr, index
        return None, "", None

    @classmethod
    def supports(cls, node: ast.expr) -> bool:
        if input_source(node) is not None:
            return True
        if isinstance(node, ast.Constant):
            return node.value is None or isinstance(node.value, (bool, int, float, str))
        if isinstance(node, ast.Attribute):
            return (
                (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "torch"
                    and isinstance(getattr(torch, node.attr, None), torch.dtype)
                )
                or node.attr == "_base"
                or (isinstance(node.value, ast.Attribute) and cls.supports(node.value))
            )
        if isinstance(node, ast.BinOp):
            return (
                type(node.op) in cls.scalar_ops
                and cls.supports(node.left)
                and cls.supports(node.right)
            )
        if isinstance(node, ast.BoolOp):
            return all(cls.supports(value) for value in node.values)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)) and cls.supports(
                node.operand
            )
        if isinstance(node, ast.IfExp):
            return all(
                cls.supports(value) for value in (node.test, node.body, node.orelse)
            )
        if isinstance(node, ast.Compare):
            return cls._supports_comparison(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return cls._supports_call(node)
        return cls._supports_metadata(node)


@dataclass(frozen=True)
class _Expression:
    """Static expression facts; operands stop at the first whole-guard fallback."""

    type: ir.Type
    operands: tuple[ast.expr, ...] = ()
    pure: bool = True
    fallback: bool | None = None
    array_size: bool = False


class GuardExpressionVisitor:
    """Build complete expression Results, with static whole-expression fallback metadata."""

    def __init__(
        self,
        text: str,
        scope: Scope | None,
        context: ir.Context,
    ) -> None:
        self.text = text
        self.scope = scope
        self.context = context
        self._analysis: dict[ast.expr, _Expression] = {}

    def _analyze(self, node: ast.expr) -> _Expression:
        """Analyze syntax once without materializing inputs or creating IR."""
        if node not in self._analysis:
            self._analysis[node] = self._analyze_node(node)
        return self._analysis[node]

    def _analyze_node(self, node: ast.expr) -> _Expression:
        source = input_source(node)
        if source is not None:
            return self._analyze_source(source)
        if isinstance(node, ast.Constant):
            return _Expression(value_type(node.value))
        if self._direct_fallback(node):
            return _Expression(tvm_ffi_d.BoolType.get(), fallback=True)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return _Expression(value_type(getattr(torch, node.attr)))
        if isinstance(node, ast.Call) and ASTVisitor.device(node) is not None:
            return _Expression(tvm_ffi_d.DeviceType.get())

        array_size = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and not (
                isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "shape"
            )
        )
        operands: list[ast.expr] = []
        children: list[_Expression] = []
        for operand in self._operands(node):
            operands.append(operand)
            child = self._analyze(operand)
            children.append(child)
            if child.fallback is not None:
                break
        fallback = children[-1].fallback if children else None
        return _Expression(
            self._expression_type(node, [child.type for child in children], fallback),
            tuple(operands),
            pure=not array_size and all(child.pure for child in children),
            fallback=fallback,
            array_size=array_size,
        )

    def _analyze_source(self, source: ast.Subscript) -> _Expression:
        assert self.scope is not None
        type_ = self.scope.type
        scope: Scope | None = self.scope
        for key in self.scope.resolve(source):
            scope = None if scope is None else scope.get(key)
            if scope is not None:
                type_ = scope.type
            elif isinstance(type_, tvm_ffi_d.TupleType):
                type_ = type_.types[key]
            elif isinstance(type_, tvm_ffi_d.ArrayType):
                type_ = type_.element_type
            else:
                type_ = tvm_ffi_d.AnyType.get()
        return _Expression(type_, pure=scope is not None)

    def _bind(
        self,
        stack: ScopeStack,
        types: Sequence[ir.Type],
        inputs: Sequence[ir.Value],
    ) -> BindScope:
        # Inputs are built in the parent before taking this binding snapshot.
        stack.enter_context(self._local_scope())
        return stack.push(BindScope(types, inputs))

    def _build_comparison(self, node: ast.Compare) -> list[ir.Value]:
        expression = self._analyze(node)
        operands = expression.operands
        types = make_result_types([expression.type])
        with ScopeStack() as stack:
            comparisons: list[ir.Value] = []
            for index, operation in enumerate(node.ops):
                pair = operands[2 * index : 2 * index + 2]
                values = self._bind(stack, types, self._build_operands(pair)).arguments
                if any(self._analyze(operand).fallback is not None for operand in pair):
                    # A whole-expression fallback replaces preceding comparisons.
                    value = values[-1]
                    break
                comparisons.append(self._compare(operation, *values))
            else:
                value = reduce(
                    BINARY_ARITHMETIC[operator.and_], comparisons, constant(True)
                )
            results = [pure(value, result_type=types[0])]
            return stack.yield_(results)

    def _build_operands(self, operands: Sequence[ast.expr]) -> list[ir.Value]:
        """Build analyzed operands, already truncated at the first fallback."""
        types = make_result_types(self._analyze(operand).type for operand in operands)

        with ScopeStack() as stack:
            values: list[ir.Value] = []
            for operand in operands:
                if self._analyze(operand).pure:
                    values.append(self.value(operand))
                else:
                    bind_scope = self._bind(stack, types, self.build(operand))
                    values.extend(bind_scope.arguments)
            results = [
                pure(value, result_type=result_type)
                for value, result_type in zip(
                    values or [constant(None)], types, strict=True
                )
            ]
            return stack.yield_(results)

    def _compare(self, operation: ast.cmpop, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if isinstance(operation, ast.Is) and all(
            isinstance(value.type, tvm_ffi_d.TensorType) for value in (lhs, rhs)
        ):
            warnings.warn(
                f"Skipping unsupported Tensor identity guard; object identity will not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return constant(True)
        return (
            equal(lhs, rhs)
            if isinstance(operation, (ast.Eq, ast.Is))
            else BINARY_ARITHMETIC[ASTVisitor.scalar_ops[type(operation)]](lhs, rhs)
        )

    def _container_size(self, value: ir.Value) -> list[ir.Value]:
        if isinstance(value.type, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)):
            value = tvm_ffi_d.downcast(
                tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get()), value
            )
        if isinstance(value.type, tvm_ffi_d.TupleType):
            # Entry guards validate untrusted tuple arity before the static
            # schema may be used; static type arity cannot validate the input.
            return checked_call("ffi.ArraySize", [value], tvm_ffi_d.IntType.get())
        return [pure(tvm_ffi_d.array_size(value))]

    def _direct_fallback(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "_base"
        if isinstance(node, (ast.Call, ast.Subscript)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return False
            source, method, _ = ASTVisitor.metadata(node)
            return source is None or method == "storage_offset"
        return False

    def _evaluate(self, node: ast.expr, values: Sequence[ir.Value]) -> ir.Value:
        if isinstance(node, ast.Attribute):
            [value] = values
            return value
        if isinstance(node, ast.BinOp):
            return BINARY_ARITHMETIC[ASTVisitor.scalar_ops[type(node.op)]](*values)
        if isinstance(node, ast.BoolOp):
            return reduce(
                BINARY_ARITHMETIC[
                    operator.and_ if isinstance(node.op, ast.And) else operator.or_
                ],
                values,
            )
        if isinstance(node, ast.IfExp):
            return select(*values)
        if isinstance(node, ast.UnaryOp):
            [value] = values
            if isinstance(node.op, ast.Not):
                return logical_not(value)
            return negate(value) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != "len":
                return BINARY_ARITHMETIC[ASTVisitor.scalar_ops[node.func.id]](*values)
            method, index = "ndimension", None
        else:
            _, method, index = ASTVisitor.metadata(node)
        [value] = values
        return self._tensor_metadata(value, method, index)

    @staticmethod
    def _expression_type(
        node: ast.expr, types: Sequence[ir.Type], fallback: bool | None
    ) -> ir.Type:
        if (
            fallback is not None
            or isinstance(node, ast.Compare)
            or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))
        ):
            return tvm_ffi_d.BoolType.get()
        if isinstance(node, (ast.Attribute, ast.BoolOp, ast.UnaryOp)):
            first_type = types[0]
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(first_type, tvm_ffi_d.BoolType)
            ):
                return tvm_ffi_d.IntType.get()
            return first_type
        if isinstance(node, ast.IfExp):
            return types[1]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
        ):
            return tvm_ffi_d.IntType.get()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return tvm_ffi_d.FloatType.get()
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.BitOr)
            and all(isinstance(type_, tvm_ffi_d.BoolType) for type_ in types)
        ):
            return tvm_ffi_d.BoolType.get()
        if any(isinstance(type_, tvm_ffi_d.FloatType) for type_ in types):
            return tvm_ffi_d.FloatType.get()
        return tvm_ffi_d.IntType.get()

    @contextmanager
    def _local_scope(self) -> Iterator[None]:
        """Inherit bindings while keeping nested IR values out of the parent."""
        scope = self.scope
        try:
            yield
        finally:
            self.scope = scope

    def _operands(self, node: ast.expr) -> Sequence[ast.expr]:
        """Select operands in evaluation order, retaining comparison diagnostics."""

        if isinstance(node, ast.BinOp):
            return (node.left, node.right)
        if isinstance(node, ast.BoolOp):
            return node.values
        if isinstance(node, ast.IfExp):
            return (node.test, node.body, node.orelse)
        if isinstance(node, ast.Compare):
            identity = any(isinstance(operation, ast.Is) for operation in node.ops)
            assert not identity or len(node.ops) == 1, (
                f"identity guard comparison must be binary: {self.text!r} in {self.context}"
            )
            assert all(
                isinstance(
                    operation,
                    (ast.Eq, ast.Gt, ast.GtE, ast.Is, ast.Lt, ast.LtE, ast.NotEq),
                )
                for operation in node.ops
            ), f"unsupported guard comparison operator: {self.text!r} in {self.context}"
            assert all(
                ASTVisitor.supports(value) for value in (node.left, *node.comparators)
            ), (
                f"guard comparison operand cannot be lowered: {self.text!r} in {self.context}"
            )
            # Preserve the existing pairwise evaluation order, including shared operands.
            return [
                value
                for operands in pairwise((node.left, *node.comparators))
                for value in operands
            ]
        if isinstance(node, ast.UnaryOp):
            if (
                isinstance(node.op, ast.Not)
                and input_source(node.operand) is not None
                and isinstance(
                    self._analyze(node.operand).type,
                    (tvm_ffi_d.ArrayType, tvm_ffi_d.TupleType, tvm_ffi_d.AnyType),
                )
            ):
                return [ast.Call(ast.Name("len", ast.Load()), [node.operand], [])]
            return [node.operand]
        if isinstance(node, ast.Attribute):
            return [node.value]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len":
                [argument] = node.args
                return [
                    argument.value
                    if isinstance(argument, ast.Attribute) and argument.attr == "shape"
                    else argument
                ]
            return node.args
        source, _, _ = ASTVisitor.metadata(node)
        return [] if source is None else [source]

    @staticmethod
    def _tensor_metadata(value: ir.Value, method: str, index: int | None) -> ir.Value:
        if not isinstance(value.type, tvm_ffi_d.TensorType):
            value = tvm_ffi_d.downcast(tvm_ffi_d.TensorType.get(), value)
        if method == "ndimension":
            return tvm_ffi_d.to(tvm_ffi_d.tensor_dim(value))
        assert index is not None
        operation = (
            tvm_ffi_d.tensor_size if method == "size" else tvm_ffi_d.tensor_stride
        )
        return tvm_ffi_d.to(operation(value, native(constant(index))))

    def _value_operands(self, node: ast.expr, operands: Sequence[ast.expr]) -> ir.Value:
        """Evaluate active operands in order, propagating whole-guard fallbacks."""
        values: list[ir.Value] = []
        comparisons: list[ir.Value] = []
        for index, operand in enumerate(operands):
            value = self.value(operand)
            values.append(value)
            if self._analyze(operand).fallback is not None:
                return value
            if isinstance(node, ast.Compare) and index % 2:
                comparisons.append(self._compare(node.ops[index // 2], *values[-2:]))
        if isinstance(node, ast.Compare):
            return reduce(BINARY_ARITHMETIC[operator.and_], comparisons, constant(True))
        return self._evaluate(node, values)

    def build(self, node: ast.expr) -> list[ir.Value]:
        expression = self._analyze(node)
        types = make_result_types([expression.type])
        if expression.pure:
            return [pure(self.value(node), result_type=types[0])]
        source = input_source(node)
        if source is not None:
            assert self.scope is not None
            result, self.scope = self.scope.extract(self.scope.resolve(source))
            return [result]
        if isinstance(node, ast.Compare):
            return self._build_comparison(node)
        bind = BindScope(types, self._build_operands(expression.operands))
        with self._local_scope(), bind as bind_scope:
            payloads = bind_scope.arguments
            if expression.fallback is not None:
                *_, fallback_value = payloads
                results = [pure(fallback_value, result_type=types[0])]
            elif expression.array_size:
                results = self._container_size(payloads[0])
            else:
                results = [pure(self._evaluate(node, payloads), result_type=types[0])]
            bind_scope.yield_(results)
            bound_results = bind_scope.results
        return [*bound_results]

    def value(self, node: ast.expr) -> ir.Value:
        """Evaluate a pure expression through ordinary values and shared operations."""
        expression = self._analyze(node)
        source = input_source(node)
        if source is not None:
            assert self.scope is not None
            value, self.scope = self.scope.read(self.scope.resolve(source))
            return value
        if isinstance(node, ast.Constant):
            return constant(node.value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return constant(getattr(torch, node.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            device = ASTVisitor.device(node)
            if device is not None:
                return constant(device)
        if self._direct_fallback(node):
            assert expression.fallback is not None
            source, _, _ = ASTVisitor.metadata(node)
            if source is None:
                warnings.warn(
                    f"Ignoring unobservable Tensor._base guard: {self.text!r} in {self.context}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return constant(expression.fallback)
        return self._value_operands(node, expression.operands)
