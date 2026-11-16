// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERARITH_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_LOWERARITH_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_LOWERARITH
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_LOWERARITH
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

/// Expand semantic scalar arithmetic into extraction, native arithmetic and
/// boxing.
void populateLowerArithPatterns(mlir::RewritePatternSet &patterns);

} // namespace trident::tvm_ffi

#endif
