# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Parse guard expressions once, without an IR context or input values."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..source import input_source, source_path

Kind = Literal[
    "binary",
    "boolean",
    "compare",
    "constant",
    "eager_boolean",
    "eager_compare",
    "eager_select",
    "input",
    "length",
    "metadata",
    "prefix",
    "select",
    "skip",
    "unary",
]


@dataclass(frozen=True)
class Expression:
    """A context-free recipe; value holds an operator, literal or input path."""

    kind: Kind
    operands: tuple[Expression, ...] = ()
    value: Any = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            return NotImplemented
        # Python equates True, 1 and 1.0, but their semantic IR types differ.
        return (
            self.kind == other.kind
            and self.operands == other.operands
            and type(self.value) is type(other.value)
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.operands, type(self.value), self.value))

    def walk(self) -> Iterator[Expression]:
        yield self
        for operand in self.operands:
            yield from operand.walk()


class UnsupportedExpression(ValueError):
    """Syntax outside the supported Dynamo guard language."""


_BINARY = {
    ast.Add: "add",
    ast.BitOr: "or",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Mult: "mul",
    ast.Sub: "sub",
}
_COMPARISON = {
    ast.Eq: "eq",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Is: "is",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.NotEq: "ne",
}
_UNARY = {ast.Not: "not", ast.UAdd: "pos", ast.USub: "neg"}


def parse_expression(node: ast.expr) -> Expression:
    source = input_source(node)
    if source is not None:
        return Expression("input", value=source_path(source))
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, float, str))
    ):
        return Expression("constant", value=node.value)
    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and isinstance(value := getattr(torch, node.attr, None), torch.dtype)
        ):
            return Expression("constant", value=value)
        if node.attr == "_base":
            return Expression("skip", value="base")
        if isinstance(node.value, ast.Attribute):
            return parse_expression(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return Expression(
            "binary",
            (parse_expression(node.left), parse_expression(node.right)),
            _BINARY[type(node.op)],
        )
    if isinstance(node, ast.BoolOp):
        return Expression(
            "boolean",
            tuple(map(parse_expression, node.values)),
            "and" if isinstance(node.op, ast.And) else "or",
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return Expression(
            "unary", (parse_expression(node.operand),), _UNARY[type(node.op)]
        )
    if isinstance(node, ast.IfExp):
        return Expression(
            "select", tuple(map(parse_expression, (node.test, node.body, node.orelse)))
        )
    if isinstance(node, ast.Compare):
        identity = any(isinstance(op, ast.Is) for op in node.ops)
        if identity and len(node.ops) != 1:
            raise UnsupportedExpression("identity guard comparison must be binary")
        if any(type(op) not in _COMPARISON for op in node.ops):
            raise UnsupportedExpression("unsupported guard comparison operator")
        if identity:
            [rhs] = node.comparators
            left, right = input_source(node.left), input_source(rhs)
            if not (isinstance(rhs, ast.Constant) and rhs.value is None) and not (
                left is not None
                and right is not None
                and source_path(left) != source_path(right)
            ):
                raise UnsupportedExpression("unsupported identity guard operands")
        return Expression(
            "compare",
            tuple(map(parse_expression, (node.left, *node.comparators))),
            tuple(_COMPARISON[type(op)] for op in node.ops),
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
                    return Expression("constant", value=torch.device(**keywords))
            except (SyntaxError, TypeError, ValueError, RuntimeError):
                pass
        if not node.keywords:
            if name in ("max", "min") and len(node.args) == 2:
                return Expression(
                    "binary", tuple(map(parse_expression, node.args)), name
                )
            if name == "len" and len(node.args) == 1:
                [argument] = node.args
                shape = isinstance(argument, ast.Attribute) and argument.attr == "shape"
                source = input_source(argument.value if shape else argument)
                if source is not None:
                    return Expression(
                        "metadata" if shape else "length",
                        (parse_expression(source),),
                        ("shape", None) if shape else None,
                    )
    index = None
    call = node
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
        source = input_source(call.func.value)
        method = call.func.attr
        if source is not None:
            if method == "storage_offset" and index is None:
                return Expression("skip", value="offset")
            if (method == "ndimension" and index is None) or (
                method in ("size", "stride") and index is not None
            ):
                return Expression(
                    "metadata", (parse_expression(source),), (method, index)
                )
        elif (
            isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "_base"
        ):
            return Expression("skip", value="base")
    raise UnsupportedExpression(f"unsupported guard syntax: {ast.unparse(node)}")
