// SPDX-License-Identifier: MIT
#ifndef TRIDENT_DIRECT_ALLOCATIONS_H_
#define TRIDENT_DIRECT_ALLOCATIONS_H_
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
namespace trident::tvm_ffi::api {
#define GEN_PASS_DECL_DIRECTALLOCATIONS
#define GEN_PASS_DECL_FLATTENDIRECTALLOCATIONS
#define GEN_PASS_DECL_UNBOXSCFIFTUPLES
#include "trident/core/Dialect/TVMFFI/API/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION_DIRECTALLOCATIONS
#define GEN_PASS_REGISTRATION_FLATTENDIRECTALLOCATIONS
#define GEN_PASS_REGISTRATION_UNBOXSCFIFTUPLES
#include "trident/core/Dialect/TVMFFI/API/Transforms/Passes.h.inc"
} // namespace trident::tvm_ffi::api
#endif
