# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Parse guard expressions once, without an IR context or input values."""

from __future__ import annotations

import ast

import torch

from trident.utils.guards import input_path

from .nodes import (
    BaseExpression,
    BinaryExpression,
    BinaryOperator,
    BooleanExpression,
    ComparisonExpression,
    ComparisonOperator,
    ConstantExpression,
    Expression,
    InputExpression,
    LengthExpression,
    RankExpression,
    SelectExpression,
    SizeExpression,
    StorageOffsetExpression,
    StrideExpression,
    UnaryExpression,
    UnaryOperator,
    UnsupportedExpression,
)

_BINARY: dict[type[ast.operator], BinaryOperator] = {
    ast.Add: "add",
    ast.BitOr: "or",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Mult: "mul",
    ast.Sub: "sub",
}
_COMPARISON: dict[type[ast.cmpop], ComparisonOperator] = {
    ast.Eq: "eq",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Is: "is",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.NotEq: "ne",
}
_UNARY: dict[type[ast.unaryop], UnaryOperator] = {
    ast.Not: "not",
    ast.UAdd: "pos",
    ast.USub: "neg",
}


def parse_expression(node: ast.expr) -> Expression:
    """Construct a complete recipe without an IR context or input values."""
    if (path := input_path(node)) is not None:
        return InputExpression(path)
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, float, str))
    ):
        return ConstantExpression(node.value)
    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and isinstance(value := getattr(torch, node.attr, None), torch.dtype)
        ):
            return ConstantExpression(value)
        if node.attr == "_base":
            return BaseExpression(parse_expression(node.value))
        if isinstance(node.value, ast.Attribute):
            return parse_expression(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        lhs, rhs = map(parse_expression, (node.left, node.right))
        return BinaryExpression(lhs, rhs, _BINARY[type(node.op)])
    if isinstance(node, ast.BoolOp):
        return BooleanExpression(
            tuple(map(parse_expression, node.values)),
            "and" if isinstance(node.op, ast.And) else "or",
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        operand = parse_expression(node.operand)
        return UnaryExpression(operand, _UNARY[type(node.op)])
    if isinstance(node, ast.IfExp):
        return SelectExpression(
            *map(parse_expression, (node.test, node.body, node.orelse))
        )
    if isinstance(node, ast.Compare):
        identity = any(isinstance(op, ast.Is) for op in node.ops)
        if identity and len(node.ops) != 1:
            raise UnsupportedExpression("identity guard comparison must be binary")
        if any(type(op) not in _COMPARISON for op in node.ops):
            raise UnsupportedExpression("unsupported guard comparison operator")
        if identity:
            [rhs] = node.comparators
            left, right = input_path(node.left), input_path(rhs)
            if not (isinstance(rhs, ast.Constant) and rhs.value is None) and not (
                left is not None and right is not None and left != right
            ):
                raise UnsupportedExpression("unsupported identity guard operands")

        return ComparisonExpression(
            tuple(map(parse_expression, (node.left, *node.comparators))),
            tuple(_COMPARISON[type(op)] for op in node.ops),
            text=ast.unparse(node),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "device" and not node.args:
            try:
                keywords = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                }
                if len(keywords) == len(node.keywords) and None not in keywords:
                    return ConstantExpression(torch.device(**keywords))
            except (SyntaxError, TypeError, ValueError, RuntimeError):
                pass
        if not node.keywords:
            if name in ("max", "min") and len(node.args) == 2:
                lhs, rhs = map(parse_expression, node.args)
                return BinaryExpression(lhs, rhs, "max" if name == "max" else "min")
            if name == "len" and len(node.args) == 1:
                [argument] = node.args
                shape = isinstance(argument, ast.Attribute) and argument.attr == "shape"
                if (
                    path := input_path(argument.value if shape else argument)
                ) is not None:
                    operand = InputExpression(path)
                    return (
                        RankExpression(operand) if shape else LengthExpression(operand)
                    )
    index: int | None = None
    call: ast.expr = node
    if isinstance(node, ast.Subscript):
        if (
            not isinstance(node.slice, ast.Constant)
            or type(node.slice.value) is not int
        ):
            raise UnsupportedExpression(
                "tensor metadata index must be a constant integer"
            )
        index, call = node.slice.value, node.value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and not call.args
        and not call.keywords
    ):
        receiver = call.func.value
        method = call.func.attr
        if input_path(receiver) is not None or (
            isinstance(receiver, ast.Attribute) and receiver.attr == "_base"
        ):
            if method == "storage_offset" and index is None:
                return StorageOffsetExpression(parse_expression(receiver))
            if method == "ndimension" and index is None:
                return RankExpression(parse_expression(receiver))
            if method in ("size", "stride") and index is not None:
                operand = parse_expression(receiver)
                return (
                    SizeExpression(operand, index)
                    if method == "size"
                    else StrideExpression(operand, index)
                )
    raise UnsupportedExpression(f"unsupported guard syntax: {ast.unparse(node)}")
