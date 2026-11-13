# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import Final

from typing_extensions import Self, override

from trident.core import ir

from ..expressions import (
    Expression,
    ExpressionBuilder,
    UnsupportedExpression,
    parse_expression,
)
from ..scope import Scope
from ..source import input_source
from .base import GuardCode


class ASTCode(GuardCode):
    def __init__(
        self,
        text: str,
        expression: ast.expr,
        *,
        source: ast.Subscript | None = None,
    ) -> None:
        super().__init__(text, source)
        self.expression: Final[Expression] = parse_expression(expression)
        self._expression_key: Final[str] = ast.dump(
            expression, include_attributes=False
        )
        # Preserve early constant checks independently of the Dynamo category.
        self.priority = 2
        if (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and input_source(expression.left) is not None
        ):
            [operation], [comparator] = expression.ops, expression.comparators
            if (
                isinstance(operation, ast.Eq)
                and input_source(comparator) is None
                or isinstance(operation, ast.Is)
                and isinstance(comparator, ast.Constant)
                and comparator.value is None
            ):
                self.priority = 1

    @override
    def build_result(self, scope: Scope, context: ir.Context) -> list[ir.Value]:
        return [
            ExpressionBuilder(self.text, scope, context).build(
                self.expression, predicate=True
            )
        ]

    @property
    def key(self) -> Hashable:
        return ("ast", self._expression_key)

    @classmethod
    def parse(
        cls,
        text: str,
        source: ast.Subscript | None,
    ) -> Self | None:
        try:
            expression = ast.parse(text.strip(), mode="eval").body
            return cls(text, expression, source=source)
        except (SyntaxError, UnsupportedExpression):
            return None
