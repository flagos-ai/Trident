//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_BASE_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_BASE_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeBase.td"

def TVMFFI_BoolType
    : TVMFFIStaticTypeIndex_Type<"Bool", "bool", 2,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
def TVMFFI_C_AnyType
    : TVMFFICBaseABI_Type<"CAny", "c.any", [TVMFFI_Contains]>;
def TVMFFI_C_ArrayType
    : TVMFFIStaticTypeIndex_Type<"CArray", "c.array", 71,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_C_ExceptionType
    : TVMFFICBaseABI_Type<"CException", "c.exception",
                         [TVMFFI_HasObject, TVMFFI_TypeIndexInterface], [{
    static int32_t getTypeIndex();
  }]>;
def TVMFFI_C_FunctionType
    : TVMFFIStaticTypeIndex_Type<"CFunction", "c.function", 68,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_C_StrType
    : TVMFFIStaticTypeIndex_Type<"CStr", "c.str", 65,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>,
                          TVMFFI_HasObject]>;
def TVMFFI_C_TensorType
    : TVMFFIStaticTypeIndex_Type<"CTensor", "c.tensor", 70,
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
def TVMFFI_FloatType
    : TVMFFIStaticTypeIndex_Type<"Float", "float", 3,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
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
def TVMFFI_RawStrType : TVMFFIStaticTypeIndex_Type<"RawStr", "raw_str", 8>;
def TVMFFI_SmallStrType
    : TVMFFIStaticTypeIndex_Type<"SmallStr", "small_str", 11>;

#endif
