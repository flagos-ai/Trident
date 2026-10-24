//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

/// Register default type conversions, patterns and target legality.
/// Keep the converter at a stable address through pattern application. Add any
/// overriding type rules after population. After successful conversion, replace
/// nested type attributes as in the pass.
void populateDecomposeTVMFFIConversionPatterns(
    mlir::ConversionTarget &target, mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_
