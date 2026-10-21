# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Lexical visibility queries for frontend SSA values."""

from trident.core import ir


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
