// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_C_TVMFFIMONADTYPES_H
#define TRIDENT_CORE_C_TVMFFIMONADTYPES_H
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#ifdef __cplusplus
extern "C" {
#endif
// ResultType
// The payload must be a TVM FFI C ABI type; use None for no-data results.

MLIR_CAPI_EXPORTED MlirType
mlirTVMFFIMonadResultTypeGetChecked(MlirLocation location, MlirType payload);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIMonadResultTypeGetName(void);
MLIR_CAPI_EXPORTED MlirType
mlirTVMFFIMonadResultTypeGetPayloadType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIMonadResultTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIMonadResultType(MlirType type);

#ifdef __cplusplus
}
#endif
#endif
