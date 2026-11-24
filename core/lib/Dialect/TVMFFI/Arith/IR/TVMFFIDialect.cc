// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIOps.h" // NOLINT(misc-include-cleaner)

#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIArithDialect.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIArith.cpp.inc"
namespace trident::tvm_ffi::arith {
void TVMFFIArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIArith.cpp.inc"
      >();
}
} // namespace trident::tvm_ffi::arith
