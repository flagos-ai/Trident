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


class ASTCode(GuardCode):
    priority = 2

    def __init__(
        self,
        text: str,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, None)
        self.expression: Final[ast.expr] = expression
        self.supported: Final[bool] = ASTVisitor.supports(expression)

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        """Build standalone expressions used by the guard IR tests."""
        from trident.importer import TridentFxImporter

        importer = TridentFxImporter(context=context, specialization_id=0)
        return GuardExpressionVisitor(self.text, tree, context, importer).value(
            self.expression
        )

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
