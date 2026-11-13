# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from ast import Subscript

from .ast import ASTCode as ASTCode
from .base import GuardCode as GuardCode
from .tensor import DynamoAttributeAbsentCode as DynamoAttributeAbsentCode
from .tensor import RequiresGradCode as RequiresGradCode
from .tensor import TensorDeviceCode as TensorDeviceCode
from .tensor import TensorDTypeCode as TensorDTypeCode
from .tensor import TensorRankCode as TensorRankCode
from .type import TypeIdCode as TypeIdCode


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
