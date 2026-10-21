//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_OPAQUE_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_OPAQUE_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeBase.td"

def TVMFFI_AnyType
    : TVMFFICOpaqueABI_Type<"Any", "any",
                         [TVMFFI_Contains]>;
def TVMFFI_ArrayType
    : TVMFFICOpaqueABI_Type<"Array", "array"> {
  let summary = "An array whose elements are TVM FFI C ABI values";
  let parameters = (ins "::trident::tvm_ffi::TVMFFICABIType":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}
def TVMFFI_ExceptionType
    : TVMFFICOpaqueABI_Type<"Exception", "exception">;
def TVMFFI_FunctionType
    : TVMFFICOpaqueABI_Type<"Function", "function", [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>]>;
def TVMFFI_StrType
    : TVMFFICOpaqueABI_Type<"Str", "str">;
def TVMFFI_TensorType
    : TVMFFICOpaqueABI_Type<"Tensor", "tensor", [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>]>;
def TVMFFI_TupleType
    : TVMFFICOpaqueABI_Type<"Tuple", "tuple"> {
  let summary = "A tuple whose elements are TVM FFI C ABI types";
  let parameters = (ins ArrayRefParameter<
      "::trident::tvm_ffi::TVMFFICABIType">:$types);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}
def TVMFFI_UnionType
    : TVMFFICOpaqueABI_Type<"Union", "union",
                         [TVMFFI_Contains]> {
  let summary = "A finite set of TVM FFI C ABI alternatives";
  let parameters = (ins ArrayRefParameter<
      "::trident::tvm_ffi::TVMFFICABIType">:$types);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif
