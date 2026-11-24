//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABIDEFER_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABIDEFER_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi::abi {

#define GEN_PASS_DECL_LOWERTVMFFIABIDEFER
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERTVMFFIABIDEFER
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/Passes.h.inc"

} // namespace trident::tvm_ffi::abi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABIDEFER_H_
