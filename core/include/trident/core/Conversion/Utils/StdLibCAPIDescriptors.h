//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_

#include "trident/core/Conversion/Utils/CFunctionDeclUtils.h"
#include <cstdlib>

// Standard C library function descriptors.
namespace trident::conversion::utils {

// Spell the ABI signature explicitly: Clang preserves abort's noreturn
// attribute in decltype(&::abort), which is not a plain function pointer type.
inline mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateAbort(mlir::ModuleOp moduleOp) {
  return getOrCreateCAPI<void (*)()>(moduleOp, "abort");
}

TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(free, Free)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
