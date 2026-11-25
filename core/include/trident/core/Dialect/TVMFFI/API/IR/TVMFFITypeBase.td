//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFITYPEBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFITYPEBASE_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/Support/TVMFFITypeTraits.td"

class TVMFFIAPI_Type<string name, string typeMnemonic, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Type">
    : TypeDef<TVMFFIAPI_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class TVMFFIAPI_ABIType<
    string name, string typeMnemonic, list<Trait> traits = [],
    string baseCppClass = "::trident::tvm_ffi::TVMFFIABIType">
    : TVMFFIAPI_Type<name, typeMnemonic, !listconcat([TVMFFI_ABI], traits),
                     baseCppClass>;

class TVMFFIAPICABI_Type<string name, string typeMnemonic,
                         list<Trait> traits = [], string extraDeclarations = "",
                         string baseCppClass =
                             "::trident::tvm_ffi::TVMFFICABIType">
    : TVMFFIAPI_ABIType<name, typeMnemonic, !listconcat([TVMFFI_CABI], traits),
                        baseCppClass> {
  let extraClassDeclaration = extraDeclarations;
}

// Types with a public C representation or object layout. Dynamic tagged
// representations do not have a single type index.
class TVMFFIAPIBase_Type<string name, string typeMnemonic,
                         list<Trait> traits = [], string extraDeclarations = "",
                         string baseCppClass =
                             "::trident::tvm_ffi::TVMFFICBaseABIType">
    : TVMFFIAPICABI_Type<name, typeMnemonic,
                         !listconcat([TVMFFI_CBaseABI], traits), "",
                         baseCppClass> {
  let extraClassDeclaration = extraDeclarations;
}

// Fixed indices are independent of object classification. This TableGen-only
// base generates the constant accessor for the common index interface.
class TVMFFIAPIStaticIndex_Type<string name, string typeMnemonic, int typeIndex,
                                list<Trait> traits = [],
                                string extraDeclarations = "">
    : TVMFFIAPIBase_Type<name, typeMnemonic,
                         !listconcat([TVMFFI_HasStaticTypeIndex,
                                      TVMFFI_TypeIndexInterface],
                                     traits)> {
  // clang-format off
  let extraClassDeclaration = [{
    static constexpr int32_t getTypeIndex() { return }] # typeIndex # [{; }
  }] # extraDeclarations;
  // clang-format on
}

// Semantic ABI types without a fixed C representation and type index.
class TVMFFIAPIOpaque_Type<
    string name, string typeMnemonic, list<Trait> traits = [],
    string extraDeclarations = "",
    string baseCppClass = "::trident::tvm_ffi::TVMFFICOpaqueABIType">
    : TVMFFIAPICABI_Type<name, typeMnemonic,
                         !listconcat([TVMFFI_COpaqueABI], traits),
                         extraDeclarations, baseCppClass>;

#endif
