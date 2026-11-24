// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIOps.h" // NOLINT(misc-include-cleaner)
#include <llvm/ADT/TypeSwitch.h>           // NOLINT(misc-include-cleaner)
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)

#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIMonadDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFITypes.cpp.inc"
#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIMonad.cpp.inc"
namespace trident::tvm_ffi::monad {
void TVMFFIMonadDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFITypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIMonad.cpp.inc"
      >();
}
} // namespace trident::tvm_ffi::monad
