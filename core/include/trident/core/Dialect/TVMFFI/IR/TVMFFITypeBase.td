//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPEBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPEBASE_TD_

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

// Types with a public C representation or object layout. Dynamic tagged
// representations do not have a single type index.
class TVMFFICBaseABI_Type<string name, string typeMnemonic,
                           list<Trait> traits = [],
                           string extraDeclarations = "",
                           string baseCppClass =
                               "::trident::tvm_ffi::TVMFFICBaseABIType">
    : TVMFFICABI_Type<name, typeMnemonic,
                     !listconcat([TVMFFI_CBaseABI], traits),
                     "", baseCppClass> {
  let extraClassDeclaration = extraDeclarations;
}

// Fixed indices are independent of object classification. This TableGen-only
// base generates the constant accessor for the common index interface.
class TVMFFIStaticTypeIndex_Type<string name, string typeMnemonic, int typeIndex,
                                  list<Trait> traits = [],
                                  string extraDeclarations = "">
    : TVMFFICBaseABI_Type<name, typeMnemonic,
                         !listconcat([TVMFFI_HasStaticTypeIndex,
                                      TVMFFI_TypeIndexInterface], traits)> {
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

#endif
