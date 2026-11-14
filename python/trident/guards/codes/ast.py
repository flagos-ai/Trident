# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import Final

from typing_extensions import Self, override

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.utils.guards import input_path
from trident.utils.monad import BindScope

from ..expressions import (
    Expression,
    ExpressionContext,
    UnsupportedExpression,
    parse_expression,
)
from ..scope import Scope
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
            and input_path(expression.left) is not None
        ):
            [operation], [comparator] = expression.ops, expression.comparators
            if (
                isinstance(operation, ast.Eq)
                and input_path(comparator) is None
                or isinstance(operation, ast.Is)
                and isinstance(comparator, ast.Constant)
                and comparator.value is None
            ):
                self.priority = 1

    @override
    def build_result(self, scope: Scope, context: ir.Context) -> list[ir.Value]:
        expression_context = ExpressionContext(self.text, scope, context)
        result = self.expression.build(expression_context)
        with expression_context.diagnostics():
            type_ = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
            with (
                expression_context.local_scope(),
                BindScope([type_], [result]) as bind_scope,
            ):
                [value] = bind_scope.arguments
                bind_scope.yield_([expression_context.truth(value)])
            return bind_scope.results

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
