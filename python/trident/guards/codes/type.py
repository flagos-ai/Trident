# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re

from typing_extensions import Self

from trident.core import ir
from trident.utils.values import constant

from ..scope import Scope
from .base import GuardCode


class TypeIdCode(GuardCode):
    """A type-id template covered by the wrapper function signature."""

    def build(self, scope: Scope, context: ir.Context) -> ir.Value:
        """The ABI type tag and Torch signature cover this guard."""
        return constant(True)

    @classmethod
    def parse(
        cls,
        text: str,
        source: ast.Subscript | None,
    ) -> Self | None:
        if source is None:
            return None
        source_regex = re.escape(ast.unparse(source))
        match = re.fullmatch(
            rf"\s*___check_type_id\(\s*{source_regex}(?:\s*\.shape)?\s*,\s*\d+\s*\)\s*,\s*"
            rf"type\s*=\s*<class\s+(?P<quote>['\"])"
            rf"(?P<type_name>[A-Za-z_][A-Za-z0-9_.]*)"
            rf"(?P=quote)>\s*",
            text,
        )
        return cls(text, source) if match is not None else None
