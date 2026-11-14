# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Context-free guard syntax and Result-based IR construction."""

from .lowering import ExpressionContext
from .nodes import Expression, UnsupportedExpression
from .syntax import parse_expression
