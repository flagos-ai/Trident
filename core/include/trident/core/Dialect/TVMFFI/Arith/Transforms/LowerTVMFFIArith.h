// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIARITH_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERTVMFFIARITH_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi::arith {

#define GEN_PASS_DECL_LOWERTVMFFIARITH
#include "trident/core/Dialect/TVMFFI/Arith/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERTVMFFIARITH
#include "trident/core/Dialect/TVMFFI/Arith/Transforms/Passes.h.inc"

/// Expand semantic scalar arithmetic into extraction, native arithmetic and
/// boxing.
void populateLowerTVMFFIArithPatterns(mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi::arith

#endif
