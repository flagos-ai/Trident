// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident-c/core/Dialects.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIDialect.h"
#include <mlir/CAPI/Registration.h>
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFIABI, tvm_ffi_abi,
                                      trident::tvm_ffi::abi::TVMFFIABIDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFIAPI, tvm_ffi_api,
                                      trident::tvm_ffi::api::TVMFFIAPIDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    TVMFFIArith, tvm_ffi_arith, trident::tvm_ffi::arith::TVMFFIArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    TVMFFIMonad, tvm_ffi_monad, trident::tvm_ffi::monad::TVMFFIMonadDialect)
