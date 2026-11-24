//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIATTRS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIInterfaces.td"

def TVMFFIAPI_ConstantSpecializationAttr
    : AttrDef<TVMFFIAPI_Dialect, "ConstantSpecialization", [
        DeclareAttrInterfaceMethods<TVMFFIAPI_SpecializationAttrInterface>
      ]> {
  let mnemonic = "constant_specialization";
  let summary = "Triton constant specialization assumption";
  let parameters = (ins "::mlir::Attribute":$value);
  let assemblyFormat = [{ `<` `value` `=` $value `>` }];
  let genVerifyDecl = 1;
}

def TVMFFIAPI_KernelSpecializationArrayAttr
    : TypedArrayAttrBase<TVMFFIAPI_SpecializationAttrInterface,
                         "array of Triton argument specializations">;

def TVMFFIAPI_VariableSpecializationAttr
    : AttrDef<TVMFFIAPI_Dialect, "VariableSpecialization", [
        DeclareAttrInterfaceMethods<TVMFFIAPI_SpecializationAttrInterface>
      ]> {
  let mnemonic = "variable_specialization";
  let summary = "Triton variable specialization assumption";
  let parameters = (ins
    "::mlir::TypeAttr":$kind,
    DefaultValuedParameter<"uint64_t", "1">:$divisibility
  );
  let assemblyFormat = [{
    `<` `kind` `=` $kind (`,` `divisibility` `=` $divisibility^)? `>`
  }];
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIATTRS_TD_
