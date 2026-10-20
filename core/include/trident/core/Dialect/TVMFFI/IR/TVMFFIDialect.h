//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Dialect.h>

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h.inc"

namespace trident::tvm_ffi {

inline constexpr llvm::StringLiteral kCleanupAttributeName = "tvm_ffi.cleanup";
inline constexpr llvm::StringLiteral kRetainAttributeName = "tvm_ffi.retain";

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIDIALECT_H_
