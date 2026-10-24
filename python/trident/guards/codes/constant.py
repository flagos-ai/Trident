# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import Self

from ..ast import ASTVisitor
from ..local import Local
from .ast import ASTCode


class ConstantCode(ASTCode):
    priority = 1

    def __init__(
        self,
        text: str,
        source: Local,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, expression, source=source)

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
