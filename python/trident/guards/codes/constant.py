# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable

from typing_extensions import Self

from ..ast import ASTVisitor
from ..scope import input_source
from .ast import ASTCode


class ConstantCode(ASTCode):
    priority = 1

    def __init__(
        self,
        text: str,
        source: ast.Subscript,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, expression, source=source)

    @property
    def key(self) -> Hashable:
        return (
            "constant",
            None if self.source is None else ast.dump(self.source),
            ast.dump(self.expression, include_attributes=False),
        )

    @classmethod
    def parse(
        cls,
        text: str,
        source: ast.Subscript | None,
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
            or ast.dump(expression.left) != ast.dump(source)
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
            and input_source(comparator) is None
            and ASTVisitor.supports(comparator)
        ):
            return cls(text, source, expression)
        else:
            return None
