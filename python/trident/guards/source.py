# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Dynamo input-source syntax, independent of runtime SSA bindings."""

import ast


def input_source(expression: ast.expr) -> ast.Subscript | None:
    """Recognize L['parameter'][index] syntax without constructing another path representation."""
    if not isinstance(expression, ast.Subscript) or not isinstance(
        expression.slice, ast.Constant
    ):
        return None
    key = expression.slice.value
    if isinstance(expression.value, ast.Name) and expression.value.id == "L":
        return expression if isinstance(key, str) and key else None
    elif (
        isinstance(key, (int, str))
        and not isinstance(key, bool)
        and input_source(expression.value) is not None
    ):
        return expression
    else:
        return None


def parse_source(text: str) -> ast.Subscript | None:
    """Resolve a Dynamo source, including a tensor's synthetic shape tuple."""
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(expression, ast.Attribute) and expression.attr == "shape":
        expression = expression.value
    return input_source(expression)


def source_path(source: ast.Subscript) -> tuple[int | str, ...]:
    """Translate a guard source to a path rooted at a validated parameter."""
    assert isinstance(source, ast.Subscript) and isinstance(source.slice, ast.Constant)
    key = source.slice.value
    if isinstance(source.value, ast.Name) and source.value.id == "L":
        input_path = (key,)
    else:
        input_path = (*source_path(source.value), key)
    return input_path
