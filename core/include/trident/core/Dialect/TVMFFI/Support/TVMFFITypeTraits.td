#ifndef TRIDENT_TVMFFI_TYPE_TRAITS_TD_
#define TRIDENT_TVMFFI_TYPE_TRAITS_TD_
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "trident/core/Dialect/TVMFFI/Support/TVMFFIInterfaces.td"

def TVMFFI_ABI : NativeTypeTrait<"TVMFFIABI">;
def TVMFFI_CABI : NativeTypeTrait<"TVMFFICABI">;
def TVMFFI_CBaseABI : NativeTypeTrait<"TVMFFICBaseABI">;
def TVMFFI_COpaqueABI : NativeTypeTrait<"TVMFFICOpaqueABI">;
// clang-format off
def TVMFFI_Contains : NativeTypeTrait<"Contains", [{
  bool contains(::trident::tvm_ffi::TVMFFICABIType type) const;
}]>;
// clang-format on
def TVMFFI_HasObject : NativeTypeTrait<"HasObject">;
def TVMFFI_HasStaticTypeIndex : NativeTypeTrait<"HasStaticTypeIndex">;
def TVMFFI_IsIntegral : NativeTypeTrait<"IsIntegral">;
def TVMFFI_IsNumeric : NativeTypeTrait<"IsNumeric">;
def TVMFFI_MayHaveObject
    : Type<Or<[CPred<"$_self.hasTrait<::mlir::TypeTrait::HasObject>()">,
               CPred<"$_self.hasTrait<::mlir::TypeTrait::Contains>()">]>,
           "TVM FFI type with HasObject or Contains">;

#endif
