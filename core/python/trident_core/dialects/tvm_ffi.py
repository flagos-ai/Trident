# ===----------------------------------------------------------------------=== #
#
# Part of the Trident project, under the MIT License.

# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

from .._mlir_libs._tvm_ffi import *
from ..ir import BlockArgumentList
from ._ods_common import _cext as _ods_cext
from ._tvm_ffi_ops_gen import *
from ._tvm_ffi_ops_gen import _Dialect


@_ods_cext.register_operation(_Dialect, replace=True)
class MonadBindOp(MonadBindOp):
    @property
    def arguments(self) -> BlockArgumentList:
        """Return one successful payload per input, including None payloads."""
        [block] = self.body.blocks
        return block.arguments
