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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.td"

def TVMFFI_ABI : NativeTypeTrait<"TVMFFIABI">;
def TVMFFI_CABI : NativeTypeTrait<"TVMFFICABI">;
def TVMFFI_CBaseABI : NativeTypeTrait<"TVMFFICBaseABI">;
def TVMFFI_COpaqueABI : NativeTypeTrait<"TVMFFICOpaqueABI">;
def TVMFFI_Contains : NativeTypeTrait<"Contains", [{
  bool contains(::trident::tvm_ffi::TVMFFICABIType type) const;
}]>;
def TVMFFI_HasObject : NativeTypeTrait<"HasObject">;
def TVMFFI_HasStaticTypeIndex : NativeTypeTrait<"HasStaticTypeIndex">;
def TVMFFI_MayHaveObject : Type<Or<[
  CPred<"$_self.hasTrait<::mlir::TypeTrait::HasObject>()">,
  CPred<"$_self.hasTrait<::mlir::TypeTrait::Contains>()">
]>, "TVM FFI type with HasObject or Contains">;

class TVMFFI_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<TVMFFI_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class TVMFFIABI_Type<string name, string typeMnemonic, list<Trait> traits = [],
                      string baseCppClass =
                          "::trident::tvm_ffi::TVMFFIABIType">
    : TVMFFI_Type<name, typeMnemonic, !listconcat([TVMFFI_ABI], traits),
                  baseCppClass>;

class TVMFFICABI_Type<string name, string typeMnemonic,
                       list<Trait> traits = [], string extraDeclarations = "",
                       string baseCppClass =
                           "::trident::tvm_ffi::TVMFFICABIType">
    : TVMFFIABI_Type<name, typeMnemonic,
                     !listconcat([TVMFFI_CABI], traits), baseCppClass> {
  let extraClassDeclaration = extraDeclarations;
}

// Types with a public C representation or object layout and a type index.
// Declare the common index accessor; concrete types supply its implementation.
class TVMFFICBaseABI_Type<string name, string typeMnemonic,
                           list<Trait> traits = [],
                           string extraDeclarations = "",
                           string baseCppClass =
                               "::trident::tvm_ffi::TVMFFICBaseABIType">
    : TVMFFICABI_Type<name, typeMnemonic,
                     !listconcat([TVMFFI_CBaseABI, TVMFFI_TypeIndexInterface],
                                 traits),
                     "", baseCppClass> {
  let extraClassDeclaration = [{
    static int32_t getTypeIndex();
  }] # extraDeclarations;
}

// Fixed indices are independent of object classification. This TableGen-only
// base generates the constant accessor for the common index interface.
class TVMFFIStaticTypeIndex_Type<string name, string typeMnemonic, int typeIndex,
                                  list<Trait> traits = [],
                                  string extraDeclarations = "">
    : TVMFFICBaseABI_Type<name, typeMnemonic,
                         !listconcat([TVMFFI_HasStaticTypeIndex], traits)> {
  let extraClassDeclaration = [{
    static constexpr int32_t getTypeIndex() { return }] # typeIndex # [{; }
  }] # extraDeclarations;
}

// Semantic ABI types without a fixed C representation and type index.
class TVMFFICOpaqueABI_Type<string name, string typeMnemonic,
                             list<Trait> traits = [],
                             string extraDeclarations = "",
                             string baseCppClass =
                                 "::trident::tvm_ffi::TVMFFICOpaqueABIType">
    : TVMFFICABI_Type<name, typeMnemonic,
                     !listconcat([TVMFFI_COpaqueABI], traits),
                     extraDeclarations, baseCppClass>;

def TVMFFI_AnyType
    : TVMFFICOpaqueABI_Type<"Any", "any",
                         [TVMFFI_Contains]>;
def TVMFFI_ArrayType
    : TVMFFICOpaqueABI_Type<"Array", "array"> {
  let summary = "An array whose elements are TVM FFI C ABI values";
  let parameters = (ins "::trident::tvm_ffi::TVMFFICABIType":$elementType);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}
def TVMFFI_BoolType
    : TVMFFIStaticTypeIndex_Type<"Bool", "bool", 2,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
def TVMFFI_C_ArrayType
    : TVMFFIStaticTypeIndex_Type<"CArray", "c.array", 71,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_DTypeType
    : TVMFFIStaticTypeIndex_Type<"DType", "dtype", 5, [], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t code,
                               int64_t bits, int64_t lanes = 1);
  }]>;
def TVMFFI_DeviceType
    : TVMFFIStaticTypeIndex_Type<"Device", "device", 6>;
def TVMFFI_ExceptionType
    : TVMFFICBaseABI_Type<"Exception", "exception", [TVMFFI_HasObject]>;
def TVMFFI_FloatType
    : TVMFFIStaticTypeIndex_Type<"Float", "float", 3,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
def TVMFFI_FunctionType
    : TVMFFIStaticTypeIndex_Type<"Function", "function", 68,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_IntType
    : TVMFFIStaticTypeIndex_Type<"Int", "int", 1,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t value);
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, ::mlir::Value value);
  }]>;
def TVMFFI_NoneType : TVMFFIStaticTypeIndex_Type<"None", "none", 0>;
def TVMFFI_ObjectType
    : TVMFFIABI_Type<"Object", "object">;
def TVMFFI_RawStrType : TVMFFIStaticTypeIndex_Type<"RawStr", "raw_str", 8>;
def TVMFFI_ResultType
    : TVMFFI_Type<"Result", "monad.result"> {
  let summary = "One payload guarded by a success bit";
  let parameters = (ins "::mlir::Type":$payloadType);
  let assemblyFormat = "`<` $payloadType `>`";
  let genVerifyDecl = 1;
}
def TVMFFI_SmallStrType
    : TVMFFIStaticTypeIndex_Type<"SmallStr", "small_str", 11>;
def TVMFFI_StrType
    : TVMFFIStaticTypeIndex_Type<"Str", "str", 65,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_TensorType
    : TVMFFIStaticTypeIndex_Type<"Tensor", "tensor", 70,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
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

def AnyTVMFFIAnyABIType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_C_ArrayType,
  TVMFFI_DTypeType, TVMFFI_DeviceType, TVMFFI_ExceptionType, TVMFFI_FloatType,
  TVMFFI_FunctionType, TVMFFI_IntType, TVMFFI_NoneType,
  TVMFFI_RawStrType, TVMFFI_SmallStrType, TVMFFI_StrType, TVMFFI_TensorType,
  TVMFFI_TupleType, TVMFFI_UnionType
], "TVM FFI value with a TVMFFIAny ABI representation">;
def AnyTVMFFICastResultType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_UnionType
], "widened TVM FFI any or union value">;
def AnyTVMFFIGetOperandType : AnyTypeOf<[
  TVMFFI_BoolType, TVMFFI_C_ArrayType, TVMFFI_FloatType, TVMFFI_FunctionType,
  TVMFFI_IntType, TVMFFI_StrType, TVMFFI_TensorType
], "TVM FFI value with a native type">;
def AnyTVMFFIGetResultType : AnyTypeOf<[
  AnyType, TVMFFI_ObjectType
], "TVM FFI get result">;
def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  TVMFFI_MayHaveObject, TVMFFI_ObjectType
], "TVM FFI object handle, object value or !tvm_ffi.any">;
def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_C_ArrayType,
  TVMFFI_DTypeType, TVMFFI_DeviceType, TVMFFI_ExceptionType, TVMFFI_FloatType,
  TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_RawStrType, TVMFFI_SmallStrType,
  TVMFFI_StrType, TVMFFI_TensorType, TVMFFI_TupleType, TVMFFI_UnionType
], "TVM FFI semantic value">;

#endif
