# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Lexical visibility and cleanup-owner queries for frontend SSA values."""

from typing import Final

from trident.core import ir

CLEANUP_ATTR: Final[str] = "tvm_ffi.cleanup"
RETAIN_ATTR: Final[str] = "tvm_ffi.retain"


def has_cleanup_owner(value: ir.Value) -> bool:
    """Whether a visible operation result has block-exit cleanup intent."""
    if not is_value_in_scope(value):
        return False
    if not isinstance(value, ir.OpResult):
        return False
    return CLEANUP_ATTR in value.owner.attributes and isinstance(
        value.owner.attributes[CLEANUP_ATTR], ir.UnitAttr
    )


def is_value_in_scope(value: ir.Value) -> bool:
    """Check lexical block ancestry before reusing a frontend SSA value."""
    owner = value.owner
    defining_block = owner if isinstance(owner, ir.Block) else owner.operation.block
    block = ir.InsertionPoint.current.block
    while block != defining_block:
        operation = block.owner.operation
        if operation.parent is None:
            return False
        block = operation.block
    return True
