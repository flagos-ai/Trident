// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LogicalResult.h>

namespace trident::tvm_ffi {

mlir::LogicalResult
ResultType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   TVMFFICABIType payload) {
  return TVMFFICABIType::classof(payload)
             ? mlir::success()
             : emitError() << "Result payload must be a TVM FFI C ABI type";
}

} // namespace trident::tvm_ffi
