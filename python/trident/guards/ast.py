# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import pairwise
from typing import Any, ClassVar

import torch

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.input import InputTable
from trident.ir import (
    BindScope,
    ScopeStack,
    binary,
    checked_call,
    constant,
    constant_type,
    equal,
    logical_not,
    make_result_types,
    negate,
    pure,
    select,
    tensor_dim,
    tensor_size,
    tensor_stride,
)

from .local import Local


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
    def metadata(node: ast.expr) -> tuple[Local | None, str, int | None]:
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
            return Local.from_expression(node.func.value), node.func.attr, index
        else:
            return None, "", None

    @classmethod
    def supports(cls, node: ast.expr) -> bool:
        if Local.from_expression(node) is not None:
            return True
        elif isinstance(node, ast.Constant):
            return node.value is None or isinstance(node.value, (bool, int, float, str))
        elif isinstance(node, ast.Attribute):
            return (
                (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "torch"
                    and isinstance(getattr(torch, node.attr, None), torch.dtype)
                )
                or node.attr == "_base"
                or (isinstance(node.value, ast.Attribute) and cls.supports(node.value))
            )
        elif isinstance(node, ast.BinOp):
            return (
                type(node.op) in cls.scalar_ops
                and cls.supports(node.left)
                and cls.supports(node.right)
            )
        elif isinstance(node, ast.BoolOp):
            return all(cls.supports(value) for value in node.values)
        elif isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)) and cls.supports(
                node.operand
            )
        elif isinstance(node, ast.IfExp):
            return all(
                cls.supports(value) for value in (node.test, node.body, node.orelse)
            )
        elif isinstance(node, ast.Compare):
            if any(isinstance(operation, ast.Is) for operation in node.ops):
                if len(node.ops) != 1:
                    return True  # Preserve the build-time invariant diagnostic.
                (rhs,) = node.comparators
                if isinstance(rhs, ast.Constant) and rhs.value is None:
                    return cls.supports(node.left)
                else:
                    left, right = (
                        Local.from_expression(node.left),
                        Local.from_expression(rhs),
                    )
                    return left is not None and right is not None and left != right
            else:
                return True  # Unsupported operators/operands receive precise diagnostics when visited.
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if cls.device(node) is not None:
                return True
            elif (
                node.func.id in ("max", "min")
                and len(node.args) == 2
                and not node.keywords
            ):
                return all(cls.supports(value) for value in node.args)
            elif node.func.id == "len" and len(node.args) == 1 and not node.keywords:
                (source,) = node.args
                return (
                    Local.from_expression(
                        source.value
                        if isinstance(source, ast.Attribute) and source.attr == "shape"
                        else source
                    )
                    is not None
                )
        source, method, index = cls.metadata(node)
        if source is not None:
            return (method in ("ndimension", "storage_offset") and index is None) or (
                method in ("size", "stride") and index is not None
            )
        else:
            call = node.value if isinstance(node, ast.Subscript) else node
            return (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "_base"
                and not call.args
                and not call.keywords
            )


class GuardExpressionVisitor:
    """Build complete expression Results, with static whole-expression fallback metadata."""

    def __init__(
        self,
        text: str,
        tree: InputTable | None,
        context: ir.Context,
    ) -> None:
        self.text = text
        self.tree = tree
        self.context = context
        self._analysis: dict[ast.expr, _ExpressionAnalysis] = {}

    def _analyze(self, node: ast.expr) -> _ExpressionAnalysis:
        if node in self._analysis:
            return self._analysis[node]
        source = Local.from_expression(node)
        if source is not None:
            assert self.tree is not None
            entry = self.tree.node(source.subscripts)
            assert entry is not None, f"guard source cannot be resolved: {self.text!r}"
            analysis = _ExpressionAnalysis(entry.type, pure=entry.value is not None)
        elif isinstance(node, ast.Constant):
            analysis = _ExpressionAnalysis(constant_type(node.value))
        elif self._direct_fallback(node):
            source, _, _ = ASTVisitor.metadata(node)
            analysis = _ExpressionAnalysis(
                tvm_ffi_d.BoolType.get(), fallback=source is not None
            )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            analysis = _ExpressionAnalysis(constant_type(getattr(torch, node.attr)))
        elif isinstance(node, ast.Call) and ASTVisitor.device(node) is not None:
            analysis = _ExpressionAnalysis(tvm_ffi_d.DeviceType.get())
        else:
            length = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
            )
            array_size = False
            if length:
                [argument] = node.args
                array_size = not (
                    isinstance(argument, ast.Attribute) and argument.attr == "shape"
                )
            operands: list[ast.expr] = []
            types: list[ir.Type] = []
            pure = not array_size
            fallback: bool | None = None
            for operand in self._operands(node):
                operands.append(operand)
                child = self._analyze(operand)
                types.append(child.type)
                pure = pure and child.pure
                fallback = child.fallback
                if fallback is not None:
                    break
            if (
                fallback is not None
                or isinstance(node, ast.Compare)
                or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))
            ):
                type_ = tvm_ffi_d.BoolType.get()
            elif isinstance(node, ast.Attribute):
                [type_] = types
            elif isinstance(node, (ast.BoolOp, ast.UnaryOp)):
                first_type, *_ = types
                type_ = (
                    tvm_ffi_d.IntType.get()
                    if isinstance(node, ast.UnaryOp)
                    and isinstance(node.op, ast.USub)
                    and isinstance(first_type, tvm_ffi_d.BoolType)
                    else first_type
                )
            elif isinstance(node, ast.IfExp):
                _, body_type, _ = types
                type_ = body_type
            elif length:
                type_ = tvm_ffi_d.IntType.get()
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                type_ = tvm_ffi_d.FloatType.get()
            else:
                type_ = (
                    tvm_ffi_d.BoolType.get()
                    if isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.BitOr)
                    and all(isinstance(type_, tvm_ffi_d.BoolType) for type_ in types)
                    else tvm_ffi_d.FloatType.get()
                    if any(isinstance(type_, tvm_ffi_d.FloatType) for type_ in types)
                    else tvm_ffi_d.IntType.get()
                )
            analysis = _ExpressionAnalysis(
                type_, tuple(operands), pure, fallback, array_size
            )
        self._analysis[node] = analysis
        return analysis

    def _build_comparison(self, node: ast.Compare) -> list[ir.Value]:
        analysis = self._analyze(node)
        types = make_result_types([analysis.type])
        with ScopeStack() as stack:
            comparisons: list[ir.Value] = []
            for index, operation in enumerate(node.ops):
                operands = analysis.operands[2 * index : 2 * index + 2]
                values = stack.bind(types, self._build_operands(operands))
                if any(
                    self._analyze(operand).fallback is not None for operand in operands
                ):
                    # A whole-expression fallback replaces preceding comparisons.
                    value = values[-1]
                    break
                comparisons.append(self._compare(operation, *values))
            else:
                value = reduce(
                    partial(binary, operator.and_), comparisons, constant(True)
                )
            results = pure([value], result_types=types)
            return stack.yield_(results)

    def _build_operands(self, operands: Sequence[ast.expr]) -> list[ir.Value]:
        """Build analyzed operands, already truncated at the first fallback."""
        types = make_result_types(self._analyze(value).type for value in operands)

        with ScopeStack() as stack:
            values: list[ir.Value] = []
            for operand in operands:
                if self._analyze(operand).pure:
                    values.append(self.value(operand))
                else:
                    scope = stack.push(BindScope(types, self.build(operand)))
                    values.extend(scope.arguments)
            results = pure(values, result_types=types)
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
        else:
            return (
                equal(lhs, rhs)
                if isinstance(operation, (ast.Eq, ast.Is))
                else binary(ASTVisitor.scalar_ops[type(operation)], lhs, rhs)
            )

    def _container_size(self, value: ir.Value) -> list[ir.Value]:
        if isinstance(value.type, tvm_ffi_d.TupleType):
            # Entry guards validate untrusted tuple arity before the static
            # schema may be used; static type arity cannot validate the input.
            return checked_call("ffi.ArraySize", [value], tvm_ffi_d.IntType.get())
        return pure([tvm_ffi_d.array_size(value)])

    def _direct_fallback(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "_base"
        elif isinstance(node, (ast.Call, ast.Subscript)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return False
            else:
                source, method, _ = ASTVisitor.metadata(node)
                return source is None or method == "storage_offset"
        else:
            return False

    def _evaluate(self, node: ast.expr, values: Sequence[ir.Value]) -> ir.Value:
        if isinstance(node, ast.Attribute):
            [value] = values
            return value
        elif isinstance(node, ast.BinOp):
            return binary(ASTVisitor.scalar_ops[type(node.op)], *values)
        elif isinstance(node, ast.BoolOp):
            return reduce(
                partial(
                    binary,
                    operator.and_ if isinstance(node.op, ast.And) else operator.or_,
                ),
                values,
            )
        elif isinstance(node, ast.IfExp):
            return select(*values)
        elif isinstance(node, ast.UnaryOp):
            [value] = values
            if isinstance(node.op, ast.Not):
                return logical_not(value)
            elif isinstance(node.op, ast.USub):
                return negate(value)
            else:
                return value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len":
                [value] = values
                return tensor_dim(value)
            else:
                return binary(ASTVisitor.scalar_ops[node.func.id], *values)
        else:
            _, method, index = ASTVisitor.metadata(node)
            [value] = values
            if method == "ndimension":
                return tensor_dim(value)
            else:
                assert index is not None
                return (tensor_size if method == "size" else tensor_stride)(
                    value, constant(index)
                )

    def _operands(self, node: ast.expr) -> Sequence[ast.expr]:
        """Select operands in evaluation order, retaining comparison diagnostics."""

        if isinstance(node, ast.BinOp):
            return (node.left, node.right)
        elif isinstance(node, ast.BoolOp):
            return node.values
        elif isinstance(node, ast.IfExp):
            return (node.test, node.body, node.orelse)
        elif isinstance(node, ast.Compare):
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
        elif isinstance(node, ast.UnaryOp):
            if (
                isinstance(node.op, ast.Not)
                and Local.from_expression(node.operand) is not None
            ):
                return [ast.Call(ast.Name("len", ast.Load()), [node.operand], [])]
            else:
                return [node.operand]
        elif isinstance(node, ast.Attribute):
            return [node.value]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len":
                [argument] = node.args
                return [
                    argument.value
                    if isinstance(argument, ast.Attribute) and argument.attr == "shape"
                    else argument
                ]
            else:
                return node.args
        else:
            source, _, _ = ASTVisitor.metadata(node)
            return (
                [] if source is None else [ast.parse(source.literal, mode="eval").body]
            )

    def build(self, node: ast.expr) -> list[ir.Value]:
        analysis = self._analyze(node)
        types = make_result_types([analysis.type])
        if analysis.pure:
            return pure([self.value(node)], result_types=types)
        source = Local.from_expression(node)
        if source is not None:
            assert self.tree is not None
            return self.tree.extract(source.subscripts)
        elif isinstance(node, ast.Compare):
            return self._build_comparison(node)
        else:
            with BindScope(types, self._build_operands(analysis.operands)) as scope:
                payloads = scope.arguments
                if analysis.fallback is not None:
                    *_, fallback = payloads
                    results = pure([fallback], result_types=types)
                elif analysis.array_size:
                    results = self._container_size(payloads[0])
                else:
                    results = pure([self._evaluate(node, payloads)], result_types=types)
                scope.yield_(results)
                bound_results = scope.results
            return [*bound_results]

    def value(self, node: ast.expr) -> ir.Value:
        """Evaluate a pure expression through ordinary values and shared operations."""
        analysis = self._analyze(node)
        source = Local.from_expression(node)
        if source is not None:
            assert self.tree is not None
            value = source.resolve(self.tree)
            assert value is not None
            return value
        elif isinstance(node, ast.Constant):
            return constant(node.value)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return constant(getattr(torch, node.attr))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            device = ASTVisitor.device(node)
            if device is not None:
                return constant(device)
        if self._direct_fallback(node):
            fallback = analysis.fallback
            assert fallback is not None
            if not fallback:
                warnings.warn(
                    f"Unsupported Tensor._base guard forces a specialization miss: {self.text!r} in {self.context}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return constant(fallback)
        values = []
        comparisons: list[ir.Value] = []
        for index, operand in enumerate(analysis.operands):
            value = self.value(operand)
            values.append(value)
            if self._analyze(operand).fallback is not None:
                return value
            if isinstance(node, ast.Compare) and index % 2:
                comparisons.append(self._compare(node.ops[index // 2], *values[-2:]))
        if isinstance(node, ast.Compare):
            return reduce(partial(binary, operator.and_), comparisons, constant(True))
        else:
            return self._evaluate(node, values)


@dataclass(frozen=True)
class _ExpressionAnalysis:
    type: ir.Type
    operands: tuple[ast.expr, ...] = ()
    pure: bool = True
    fallback: bool | None = None
    array_size: bool = False
