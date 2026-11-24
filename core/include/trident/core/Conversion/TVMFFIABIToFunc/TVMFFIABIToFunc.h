//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFIABITOFUNC_TVMFFIABITOFUNC_H_
#define TRIDENT_CORE_CONVERSION_TVMFFIABITOFUNC_TVMFFIABITOFUNC_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTTVMFFIABITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFIABITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

/// Register conversion patterns and legality for packed function definitions
/// and local calls.
void populateTVMFFIABIToFuncConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns);

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_TVMFFIABITOFUNC_TVMFFIABITOFUNC_H_
