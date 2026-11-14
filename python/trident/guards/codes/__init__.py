# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from ast import Subscript

from .ast import ASTCode
from .base import GuardCode
from .tensor import (
    DynamoAttributeAbsentCode,
    RequiresGradCode,
    TensorDTypeCode,
    TensorDeviceCode,
    TensorRankCode,
)
from .type import TypeIdCode


def parse_code(text: str, source: Subscript | None) -> GuardCode | None:
    """Prefer specialized regex templates, then parse general AST expressions."""
    for code_type in (
        DynamoAttributeAbsentCode,
        RequiresGradCode,
        TensorDTypeCode,
        TensorDeviceCode,
        TensorRankCode,
        TypeIdCode,
    ):
        code = code_type.parse(text, source)
        if code is not None:
            return code
    return ASTCode.parse(text, source)
