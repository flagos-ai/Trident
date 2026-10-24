//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTENSORMETADATA_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTENSORMETADATA_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_LOWERTENSORMETADATA
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERTENSORMETADATA
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

/// Register rewrites from TVM FFI tensor metadata to DLPack metadata.
void populateLowerTensorMetadataPatterns(mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTENSORMETADATA_H_
