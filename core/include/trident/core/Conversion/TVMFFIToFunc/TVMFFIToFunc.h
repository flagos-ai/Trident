//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

/// Register conversion patterns and legality for the packed function boundary.
void populateTVMFFIToFuncConversionPatterns(mlir::ConversionTarget &target,
                                            mlir::RewritePatternSet &patterns);

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_
