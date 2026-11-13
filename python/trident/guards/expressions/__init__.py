# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Context-free guard syntax and Result-based IR construction."""

from .lowering import ExpressionContext as ExpressionContext
from .nodes import Expression as Expression
from .syntax import UnsupportedExpression as UnsupportedExpression
from .syntax import parse_expression as parse_expression
