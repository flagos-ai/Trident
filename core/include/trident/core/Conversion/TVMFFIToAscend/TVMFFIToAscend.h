//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOASCEND_TVMFFITOASCEND_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOASCEND_TVMFFITOASCEND_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTTVMFFITOASCEND
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOASCEND
#include "trident/core/Conversion/Passes.h.inc"

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOASCEND_TVMFFITOASCEND_H_
