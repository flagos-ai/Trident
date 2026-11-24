// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#pragma once
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFIABI, tvm_ffi_abi);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFIAPI, tvm_ffi_api);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFIArith, tvm_ffi_arith);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFIMonad, tvm_ffi_monad);
#ifdef __cplusplus
}
#endif
