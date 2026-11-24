// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_C_TVMFFIAPITYPES_H
#define TRIDENT_CORE_C_TVMFFIAPITYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// AnyType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIAnyTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIAnyTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIAnyTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIAnyType(MlirType type);

// ArrayType

MLIR_CAPI_EXPORTED MlirType
mlirTVMFFIAPIArrayTypeGetChecked(MlirLocation location, MlirType element);
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIArrayTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIArrayTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIArrayTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIArrayType(MlirType type);

// ExceptionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIExceptionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIExceptionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIExceptionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIExceptionType(MlirType type);

// FunctionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIFunctionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIFunctionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIFunctionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIFunctionType(MlirType type);

// StrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIStrType(MlirType type);

// TensorType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPITensorTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPITensorTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPITensorTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPITensorType(MlirType type);

// TupleType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPITupleTypeGetChecked(
    MlirLocation location, intptr_t numTypes, const MlirType *types);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPITupleTypeGetName(void);
MLIR_CAPI_EXPORTED intptr_t mlirTVMFFIAPITupleTypeGetNumTypes(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPITupleTypeGetType(MlirType type,
                                                          intptr_t index);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPITupleTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPITupleType(MlirType type);

// UnionType

/// Construct a verified union, emitting diagnostics and returning a null type
/// if the members are invalid or belong to another context.
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIUnionTypeGetChecked(
    MlirLocation location, intptr_t numTypes, const MlirType *types);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIAPIUnionTypeGetName(void);
MLIR_CAPI_EXPORTED intptr_t mlirTVMFFIAPIUnionTypeGetNumTypes(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTVMFFIAPIUnionTypeGetType(MlirType type,
                                                          intptr_t index);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIAPIUnionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIAPIUnionType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif
