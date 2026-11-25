#ifndef TRIDENT_CORE_DIALECT_TVMFFI_SUPPORT_INTERFACES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_SUPPORT_INTERFACES_TD_

// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/IR/OpBase.td"

def TVMFFI_NativeTypeInterface : TypeInterface<"TVMFFINativeTypeInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  // clang-format off
  let description = [{
    Provides the native MLIR type used when a TVM FFI value is passed to a
    kernel.
  }];
  // clang-format on
  let methods = [InterfaceMethod<"Returns the corresponding native MLIR type.",
                                 "::mlir::Type", "getNativeType", (ins)>];
}

def TVMFFI_TypeIndexInterface : TypeInterface<"TVMFFITypeIndexInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  // clang-format off
  let description = [{
    Provides the concrete TVM FFI runtime type index for a semantic type.
  }];
  // clang-format on
  let methods = [StaticInterfaceMethod<
      "Returns the concrete TVM FFI runtime type index.", "int32_t",
      "getTypeIndex", (ins)>];
}

def TVMFFI_ResultTypeInterface : TypeInterface<"TVMFFIResultTypeInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  // clang-format off
  let description = [{ Exposes the C ABI payload carried by a Result type. }];
  // clang-format on
  let methods = [InterfaceMethod<"Return the guarded payload type.",
                                 "::mlir::Type", "getPayloadType", (ins)>];
}

def TVMFFI_ResultType : Type<CPred<"::llvm::isa<::trident::tvm_ffi::"
                                   "TVMFFIResultTypeInterface>($_self)">,
                             "TVM FFI Result">;

#endif
