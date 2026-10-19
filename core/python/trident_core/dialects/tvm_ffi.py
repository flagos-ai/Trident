# ===----------------------------------------------------------------------=== #
#
# Part of the Trident project, under the MIT License.

# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence

from .._mlir_libs._tvm_ffi import *
from ..ir import Block, BlockArgumentList, Location
from ._ods_common import _cext as _ods_cext
from ._tvm_ffi_ops_gen import *
from ._tvm_ffi_ops_gen import _Dialect


@_ods_cext.register_operation(_Dialect, replace=True)
class MonadBindOp(MonadBindOp):
    def add_entry_block(self, arg_locs: Sequence[Location] | None = None) -> Block:
        """Return the entry block, creating its payload arguments if needed."""
        if (block := self.get_entry_block()) is not None:
            return block
        self.body.blocks.append(
            *(value.type.payload_type for value in self.inputs), arg_locs=arg_locs
        )
        return self.body.blocks[0]

    def get_entry_block(self) -> Block | None:
        """Return the entry block, or None when the body is empty."""
        return self.body.blocks[0] if self.body.blocks else None

    @property
    def arguments(self) -> BlockArgumentList:
        """Return one successful payload per input, including None payloads."""
        block = self.get_entry_block()
        assert block is not None, "bind has no entry block"
        return block.arguments
