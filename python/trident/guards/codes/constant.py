# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import TYPE_CHECKING, Final, Self, override

from trident.core import ir
from trident.input import InputTable

from ..ast import ASTVisitor, GuardExpressionVisitor
from ..local import Local
from .base import GuardCode

if TYPE_CHECKING:
    from trident.importer import TridentFxImporter


class ConstantCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, source)
        self.expression: Final[ast.expr] = expression
        self.supported: Final[bool] = ASTVisitor.supports(expression)

    @override
    def build_result(
        self, tree: InputTable, context: ir.Context, importer: TridentFxImporter
    ) -> list[ir.Value]:
        assert self.supported, f"guard expression cannot be lowered: {self.text!r}"
        return GuardExpressionVisitor(self.text, tree, context, importer).build(
            self.expression
        )

    @property
    def key(self) -> Hashable:
        return (
            "constant",
            self.source,
            ast.dump(self.expression, include_attributes=False),
        )

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        try:
            expression = ast.parse(text.strip(), mode="eval").body
        except SyntaxError:
            return None
        if (
            not isinstance(expression, ast.Compare)
            or len(expression.ops) != 1
            or len(expression.comparators) != 1
            or Local.from_expression(expression.left) != source
        ):
            return None
        [operation] = expression.ops
        [comparator] = expression.comparators
        if (
            isinstance(operation, ast.Is)
            and isinstance(comparator, ast.Constant)
            and comparator.value is None
        ) or (
            isinstance(operation, ast.Eq)
            and Local.from_expression(comparator) is None
            and ASTVisitor.supports(comparator)
        ):
            return cls(text, source, expression)
        else:
            return None
