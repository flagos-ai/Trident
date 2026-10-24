//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTOPACK_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTOPACK_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_LOWERTOPACK
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERTOPACK
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

/// Register conversion patterns and legality for the packed function boundary.
void populateLowerToPackConversionPatterns(mlir::ConversionTarget &target,
                                           mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTOPACK_H_
