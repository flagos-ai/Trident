//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_ABI_IR_BASE_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_ABI_IR_BASE_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypeBase.td"

// Integral types carry both traits: Integral is a subset of Numeric.
def TVMFFIABI_BoolType
    : TVMFFIABIStaticIndex_Type<
          "Bool", "bool", 2,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_IsIntegral, TVMFFI_IsNumeric]>;
def TVMFFIABI_AnyType : TVMFFIABIBase_Type<"Any", "any", [TVMFFI_Contains]>;
def TVMFFIABI_ArrayType
    : TVMFFIABIStaticIndex_Type<
          "Array", "array", 71,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_HasObject]>;
def TVMFFIABI_ExceptionType
    : TVMFFIABIBase_Type<"Exception", "exception",
                         // clang-format off
                         [TVMFFI_HasObject, TVMFFI_TypeIndexInterface], [{
    static int32_t getTypeIndex();
  }]>;
// clang-format on
def TVMFFIABI_FunctionType
    : TVMFFIABIStaticIndex_Type<
          "Function", "function", 68,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_HasObject]>;
def TVMFFIABI_StrType
    : TVMFFIABIStaticIndex_Type<
          "Str", "str", 65,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_HasObject]>;
def TVMFFIABI_TensorType
    : TVMFFIABIStaticIndex_Type<
          "Tensor", "tensor", 70,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_HasObject]>;
def TVMFFIABI_DTypeType
    // clang-format off
    : TVMFFIABIStaticIndex_Type<"DType", "dtype", 5, [], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t code,
                               int64_t bits, int64_t lanes = 1);
  }]>;
// clang-format on
def TVMFFIABI_DeviceType : TVMFFIABIStaticIndex_Type<"Device", "device", 6>;
def TVMFFIABI_FloatType
    : TVMFFIABIStaticIndex_Type<
          "Float", "float", 3,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           TVMFFI_IsNumeric]>;
def TVMFFIABI_IntType
    : TVMFFIABIStaticIndex_Type<
          "Int", "int", 1,
          [DeclareTypeInterfaceMethods<TVMFFI_NativeTypeInterface>,
           // clang-format off
                          TVMFFI_IsIntegral, TVMFFI_IsNumeric], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t value);
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, ::mlir::Value value);
  }]>;
// clang-format on
def TVMFFIABI_NoneType : TVMFFIABIStaticIndex_Type<"None", "none", 0>;
def TVMFFIABI_RawStrType : TVMFFIABIStaticIndex_Type<"RawStr", "raw_str", 8>;
def TVMFFIABI_SmallStrType
    : TVMFFIABIStaticIndex_Type<"SmallStr", "small_str", 11>;

def TVMFFIABI_ObjectType : TVMFFIABI_Type<"Object", "object">;
// Borrowed pointer/count view, deliberately outside the tagged C ABI.
def TVMFFIABI_PackArgumentsType
    : TVMFFIABI_Type<"PackArguments", "pack.arguments">;

def AnyTVMFFIABIValueType
    : AnyTypeOf<[TVMFFIABI_AnyType, TVMFFIABI_ArrayType, TVMFFIABI_BoolType,
                 TVMFFIABI_DTypeType, TVMFFIABI_DeviceType,
                 TVMFFIABI_ExceptionType, TVMFFIABI_FloatType,
                 TVMFFIABI_FunctionType, TVMFFIABI_IntType, TVMFFIABI_NoneType,
                 TVMFFIABI_RawStrType, TVMFFIABI_SmallStrType,
                 TVMFFIABI_StrType, TVMFFIABI_TensorType],
                "TVM FFI ABI value">;
def AnyTVMFFIABIGetOperandType
    : AnyTypeOf<[TVMFFIABI_BoolType, TVMFFIABI_AnyType, TVMFFIABI_ArrayType,
                 TVMFFIABI_FloatType, TVMFFIABI_FunctionType, TVMFFIABI_IntType,
                 TVMFFIABI_StrType, TVMFFIABI_TensorType],
                "TVM FFI ABI value with a native type">;
def AnyTVMFFIABIGetResultType
    : AnyTypeOf<[AnyType, TVMFFIABI_ObjectType], "TVM FFI get result">;
def AnyTVMFFIABIObjectOrAnyType
    : AnyTypeOf<[TVMFFI_MayHaveObject, TVMFFIABI_ObjectType],
                "TVM FFI ABI reference">;

def TVMFFIABI_NumericType
    : Type<CPred<"$_self.hasTrait<::mlir::TypeTrait::IsNumeric>()">,
           "TVM FFI Bool, Float or Int">;
def TVMFFIABI_IntegralType
    : Type<CPred<"$_self.hasTrait<::mlir::TypeTrait::IsIntegral>()">,
           "TVM FFI Bool or Int">;
#endif
