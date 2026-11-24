//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_API_IR_OPAQUE_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_API_IR_OPAQUE_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/API/IR/TVMFFITypeBase.td"
include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.td"

def TVMFFIAPI_AnyType
    : TVMFFIAPIOpaque_Type<"Any", "any",
                         [TVMFFI_Contains]>;
def TVMFFIAPI_ArrayType
    : TVMFFIAPIOpaque_Type<"Array", "array"> {
  let summary = "An array whose elements are TVM FFI C ABI values";
  let parameters = (ins "::trident::tvm_ffi::TVMFFICABIType":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}
def TVMFFIAPI_ExceptionType
    : TVMFFIAPIOpaque_Type<"Exception", "exception">;
def TVMFFIAPI_FunctionType
    : TVMFFIAPIOpaque_Type<"Function", "function", [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>]>;
def TVMFFIAPI_StrType
    : TVMFFIAPIOpaque_Type<"Str", "str">;
def TVMFFIAPI_TensorType
    : TVMFFIAPIOpaque_Type<"Tensor", "tensor", [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>]>;
def TVMFFIAPI_TupleType
    : TVMFFIAPIOpaque_Type<"Tuple", "tuple"> {
  let summary = "A tuple whose elements are TVM FFI C ABI types";
  let parameters = (ins ArrayRefParameter<
      "::trident::tvm_ffi::TVMFFICABIType">:$types);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}
def TVMFFIAPI_UnionType
    : TVMFFIAPIOpaque_Type<"Union", "union",
                         [TVMFFI_Contains]> {
  let summary = "A finite set of TVM FFI C ABI alternatives";
  let parameters = (ins ArrayRefParameter<
      "::trident::tvm_ffi::TVMFFICABIType">:$types);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def TVMFFIAPI_AnyValue : AnyTypeOf<[TVMFFIAPI_AnyType, TVMFFIABI_AnyType], "TVM FFI dynamic tagged value"> {
  let builderCall = "::trident::tvm_ffi::api::AnyType::get($_builder.getContext())";
}

def TVMFFIAPI_ExceptionValue : AnyTypeOf<[TVMFFIABI_ExceptionType, TVMFFIAPI_ExceptionType], "TVM FFI exception value"> {
  let builderCall = "::trident::tvm_ffi::api::ExceptionType::get($_builder.getContext())";
}

def TVMFFIAPI_FunctionValue : AnyTypeOf<[TVMFFIABI_FunctionType, TVMFFIAPI_FunctionType], "TVM FFI function value"> {
  let builderCall = "::trident::tvm_ffi::api::FunctionType::get($_builder.getContext())";
}

def TVMFFIAPI_StrValue : AnyTypeOf<[TVMFFIABI_StrType, TVMFFIAPI_StrType], "TVM FFI str value"> {
  let builderCall = "::trident::tvm_ffi::api::StrType::get($_builder.getContext())";
}

def TVMFFIAPI_TensorValue : AnyTypeOf<[TVMFFIABI_TensorType, TVMFFIAPI_TensorType], "TVM FFI tensor value"> {
  let builderCall = "::trident::tvm_ffi::api::TensorType::get($_builder.getContext())";
}

def AnyTVMFFIAnyABIType : AnyTypeOf<[
  TVMFFIAPI_AnyType, TVMFFIAPI_ArrayType, TVMFFIABI_BoolType, TVMFFIABI_AnyType, TVMFFIABI_ArrayType,
  TVMFFIABI_DTypeType, TVMFFIABI_DeviceType, TVMFFIABI_ExceptionType, TVMFFIAPI_ExceptionType, TVMFFIABI_FloatType,
  TVMFFIABI_FunctionType, TVMFFIAPI_FunctionType, TVMFFIABI_IntType, TVMFFIABI_NoneType,
  TVMFFIABI_RawStrType, TVMFFIABI_SmallStrType, TVMFFIABI_StrType, TVMFFIAPI_StrType, TVMFFIABI_TensorType, TVMFFIAPI_TensorType,
  TVMFFIAPI_TupleType, TVMFFIAPI_UnionType
], "TVM FFI value with a TVMFFIAny ABI representation">;

def AnyTVMFFIGetOperandType : AnyTypeOf<[
  TVMFFIABI_BoolType, TVMFFIABI_AnyType, TVMFFIABI_ArrayType, TVMFFIABI_FloatType, TVMFFIABI_FunctionType, TVMFFIAPI_FunctionType,
  TVMFFIABI_IntType, TVMFFIABI_StrType, TVMFFIABI_TensorType, TVMFFIAPI_TensorType
], "TVM FFI value with a native type">;

def AnyTVMFFIGetResultType : AnyTypeOf<[
  AnyType, TVMFFIABI_ObjectType
], "TVM FFI get result">;

def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  TVMFFI_MayHaveObject, TVMFFIABI_ObjectType
], "TVM FFI object handle, object value or !tvm_ffi_api.any">;

def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFIAPI_AnyType, TVMFFIAPI_ArrayType, TVMFFIABI_BoolType, TVMFFIABI_AnyType, TVMFFIABI_ArrayType,
  TVMFFIABI_DTypeType, TVMFFIABI_DeviceType, TVMFFIABI_ExceptionType, TVMFFIAPI_ExceptionType, TVMFFIABI_FloatType,
  TVMFFIABI_IntType, TVMFFIABI_NoneType, TVMFFIABI_RawStrType, TVMFFIABI_SmallStrType,
  TVMFFIABI_StrType, TVMFFIAPI_StrType, TVMFFIABI_TensorType, TVMFFIAPI_TensorType, TVMFFIAPI_TupleType, TVMFFIAPI_UnionType
], "TVM FFI semantic value">;

#endif
