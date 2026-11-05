//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_

include "mlir/IR/OpBase.td"

def TVMFFI_NativeTypeInterface
    : TypeInterface<"TVMFFINativeTypeInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Provides the native MLIR type used when a TVM FFI value is passed to a
    kernel.
  }];
  let methods = [
    InterfaceMethod<
      "Returns the corresponding native MLIR type.",
      "::mlir::Type", "getNativeType", (ins)
    >
  ];
}

def TVMFFI_SpecializationAttrInterface
    : AttrInterface<"SpecializationAttrInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Describes a Triton argument specialization and materializes its runtime
    precondition when one is required.
  }];
  let methods = [
    InterfaceMethod<
      "Builds the native specialization check, or returns a null value when "
      "the specialization has no runtime precondition.",
      "::mlir::Value", "buildCheck",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
           "::mlir::Value":$operand)
    >,
    InterfaceMethod<
      "Returns the native type used to evaluate this specialization.",
      "::mlir::Type", "getTargetType", (ins)
    >
  ];
}

def TVMFFI_TypeIndexInterface
    : TypeInterface<"TVMFFITypeIndexInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Provides the concrete TVM FFI runtime type index for a semantic type.
  }];
  let methods = [
    StaticInterfaceMethod<
      "Returns the concrete TVM FFI runtime type index.",
      "int32_t", "getTypeIndex", (ins)
    >
  ];
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_
