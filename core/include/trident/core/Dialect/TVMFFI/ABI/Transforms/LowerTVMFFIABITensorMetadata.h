//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABITENSORMETADATA_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABITENSORMETADATA_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi::abi {

#define GEN_PASS_DECL_LOWERTVMFFIABITENSORMETADATA
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERTVMFFIABITENSORMETADATA
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/Passes.h.inc"

/// Register rewrites from TVM FFI tensor metadata to DLPack metadata.
void populateLowerTVMFFIABITensorMetadataPatterns(
    mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi::abi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIABITENSORMETADATA_H_
