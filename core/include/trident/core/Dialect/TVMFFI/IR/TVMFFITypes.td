//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/IR/Base/TVMFFITypes.td"
include "trident/core/Dialect/TVMFFI/IR/Monad/TVMFFITypes.td"
include "trident/core/Dialect/TVMFFI/IR/Opaque/TVMFFITypes.td"

def TVMFFI_ObjectType
    : TVMFFIABI_Type<"Object", "object">;
// Borrowed pointer/count view, deliberately outside the tagged C ABI.
def TVMFFI_PackArgumentsType
    : TVMFFI_Type<"PackArguments", "pack.arguments">;
def TVMFFI_AnyValue : AnyTypeOf<[TVMFFI_AnyType, TVMFFI_C_AnyType], "TVM FFI dynamic tagged value"> {
  let builderCall = "::trident::tvm_ffi::AnyType::get($_builder.getContext())";
}
def TVMFFI_ExceptionValue : AnyTypeOf<[TVMFFI_C_ExceptionType, TVMFFI_ExceptionType], "TVM FFI exception value"> {
  let builderCall = "::trident::tvm_ffi::ExceptionType::get($_builder.getContext())";
}
def TVMFFI_FunctionValue : AnyTypeOf<[TVMFFI_C_FunctionType, TVMFFI_FunctionType], "TVM FFI function value"> {
  let builderCall = "::trident::tvm_ffi::FunctionType::get($_builder.getContext())";
}
def TVMFFI_StrValue : AnyTypeOf<[TVMFFI_C_StrType, TVMFFI_StrType], "TVM FFI str value"> {
  let builderCall = "::trident::tvm_ffi::StrType::get($_builder.getContext())";
}
def TVMFFI_TensorValue : AnyTypeOf<[TVMFFI_C_TensorType, TVMFFI_TensorType], "TVM FFI tensor value"> {
  let builderCall = "::trident::tvm_ffi::TensorType::get($_builder.getContext())";
}

def AnyTVMFFIAnyABIType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_C_AnyType, TVMFFI_C_ArrayType,
  TVMFFI_DTypeType, TVMFFI_DeviceType, TVMFFI_C_ExceptionType, TVMFFI_ExceptionType, TVMFFI_FloatType,
  TVMFFI_C_FunctionType, TVMFFI_FunctionType, TVMFFI_IntType, TVMFFI_NoneType,
  TVMFFI_RawStrType, TVMFFI_SmallStrType, TVMFFI_C_StrType, TVMFFI_StrType, TVMFFI_C_TensorType, TVMFFI_TensorType,
  TVMFFI_TupleType, TVMFFI_UnionType
], "TVM FFI value with a TVMFFIAny ABI representation">;
def AnyTVMFFIGetOperandType : AnyTypeOf<[
  TVMFFI_BoolType, TVMFFI_C_AnyType, TVMFFI_C_ArrayType, TVMFFI_FloatType, TVMFFI_C_FunctionType, TVMFFI_FunctionType,
  TVMFFI_IntType, TVMFFI_C_StrType, TVMFFI_C_TensorType, TVMFFI_TensorType
], "TVM FFI value with a native type">;
def AnyTVMFFIGetResultType : AnyTypeOf<[
  AnyType, TVMFFI_ObjectType
], "TVM FFI get result">;
def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  TVMFFI_MayHaveObject, TVMFFI_ObjectType
], "TVM FFI object handle, object value or !tvm_ffi.any">;
def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_C_AnyType, TVMFFI_C_ArrayType,
  TVMFFI_DTypeType, TVMFFI_DeviceType, TVMFFI_C_ExceptionType, TVMFFI_ExceptionType, TVMFFI_FloatType,
  TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_RawStrType, TVMFFI_SmallStrType,
  TVMFFI_C_StrType, TVMFFI_StrType, TVMFFI_C_TensorType, TVMFFI_TensorType, TVMFFI_TupleType, TVMFFI_UnionType
], "TVM FFI semantic value">;

#endif
