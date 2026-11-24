// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_C_TVMFFIABITYPES_H
#define TRIDENT_CORE_C_TVMFFIABITYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// AnyType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIAnyTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIAnyTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIAnyTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIAnyType(MlirType type);

// ArrayType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIArrayTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIArrayTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIArrayTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIArrayType(MlirType type);

// BoolType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIBoolTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIBoolTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIBoolTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIBoolType(MlirType type);

// DTypeType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIDTypeTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIDTypeTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIDTypeTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIDTypeType(MlirType type);

// DeviceType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIDeviceTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIDeviceTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIDeviceTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIDeviceType(MlirType type);

// ExceptionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIExceptionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIExceptionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIExceptionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIExceptionType(MlirType type);

// FloatType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIFloatTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIFloatTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIFloatTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIFloatType(MlirType type);

// FunctionType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIFunctionTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIFunctionTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIFunctionTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIFunctionType(MlirType type);

// IntType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIIntTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIIntTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIIntTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIIntType(MlirType type);

// NoneType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABINoneTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABINoneTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABINoneTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABINoneType(MlirType type);

// ObjectType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIObjectTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIObjectTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIObjectTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIObjectType(MlirType type);

// PackArgumentsType

MLIR_CAPI_EXPORTED MlirType
mlirTVMFFIABIPackArgumentsTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIPackArgumentsTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIPackArgumentsTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIPackArgumentsType(MlirType type);

// RawStrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIRawStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIRawStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIRawStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIRawStrType(MlirType type);

// SmallStrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABISmallStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABISmallStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABISmallStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABISmallStrType(MlirType type);

// StrType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABIStrTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABIStrTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABIStrTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABIStrType(MlirType type);

// TensorType

MLIR_CAPI_EXPORTED MlirType mlirTVMFFIABITensorTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirStringRef mlirTVMFFIABITensorTypeGetName(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirTVMFFIABITensorTypeGetTypeID(void);
MLIR_CAPI_EXPORTED bool mlirTypeIsATVMFFIABITensorType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif
