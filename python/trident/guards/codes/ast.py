# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import Final, Self, override

from trident.core import ir
from trident.input import InputTable

from ..ast import ASTVisitor, GuardExpressionVisitor
from ..local import Local
from .base import GuardCode


class ASTCode(GuardCode):
    priority = 2

    def __init__(
        self,
        text: str,
        expression: ast.expr,
        *,
        source: Local | None = None,
    ) -> None:
        super().__init__(text, source)
        self.expression: Final[ast.expr] = expression
        self.supported: Final[bool] = ASTVisitor.supports(expression)

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        """Build standalone expressions used by the guard IR tests."""
        return GuardExpressionVisitor(self.text, tree, context).value(self.expression)

    @override
    def build_result(self, tree: InputTable, context: ir.Context) -> list[ir.Value]:
        assert self.supported, f"guard expression cannot be lowered: {self.text!r}"
        return GuardExpressionVisitor(self.text, tree, context).build(self.expression)

    @property
    def key(self) -> Hashable:
        return ("ast", ast.dump(self.expression, include_attributes=False))

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        try:
            expression = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None
        code = cls(text, expression)
        return code if code.supported else None
