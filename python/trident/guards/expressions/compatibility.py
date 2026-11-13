# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Keep historical whole-expression skips separate from ordinary evaluation."""

from itertools import pairwise

from .syntax import Expression


def compatibility_prefix(expression: Expression) -> tuple[Expression, bool]:
    """Retain eager prefixes and pairwise reads up to the first skipped item.

    The caller applies this traversal only to expressions containing a skip.
    Storage offsets remain unobserved; no metadata read is introduced here.
    """
    if expression.kind == "skip":
        return expression, True
    operands: list[Expression] = []
    if expression.kind == "compare":
        for operation, pair in zip(
            expression.value, pairwise(expression.operands), strict=True
        ):
            values: list[Expression] = []
            for operand in pair:
                child, skipped = compatibility_prefix(operand)
                values.append(child)
                if skipped:
                    return Expression("prefix", (*operands, *values)), True
            operands.append(Expression("compare", tuple(values), (operation,)))
        return Expression("eager_compare", tuple(operands)), False
    for operand in expression.operands:
        child, skipped = compatibility_prefix(operand)
        operands.append(child)
        if skipped:
            return Expression("prefix", tuple(operands)), True
    kind = {"boolean": "eager_boolean", "select": "eager_select"}.get(
        expression.kind, expression.kind
    )
    return Expression(kind, tuple(operands), expression.value), False
