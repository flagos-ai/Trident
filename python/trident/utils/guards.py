# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Dynamo input-source syntax, independent of runtime SSA bindings."""

import ast


def input_path(expression: ast.expr) -> list[int | str] | None:
    """Recognize an input access and extract its path in one traversal."""
    if not isinstance(expression, ast.Subscript) or not isinstance(
        expression.slice, ast.Constant
    ):
        return None
    key = expression.slice.value
    if isinstance(expression.value, ast.Name) and expression.value.id == "L":
        return [key] if isinstance(key, str) and key else None
    elif (
        isinstance(key, (int, str))
        and not isinstance(key, bool)
        and (path := input_path(expression.value)) is not None
    ):
        return [*path, key]
    else:
        return None
