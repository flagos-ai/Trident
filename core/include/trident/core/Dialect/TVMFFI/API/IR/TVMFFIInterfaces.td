#ifndef TRIDENT_TVMFFI_API_INTERFACES_TD_
#define TRIDENT_TVMFFI_API_INTERFACES_TD_
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/IR/OpBase.td"

def TVMFFIAPI_SpecializationAttrInterface
    : AttrInterface<"SpecializationAttrInterface"> {
  let cppNamespace = "::trident::tvm_ffi::api";
  // clang-format off
  let description = [{
    Describes a Triton argument specialization and materializes its runtime
    precondition when one is required.
  }];
  // clang-format on
  let methods =
      [InterfaceMethod<"Builds the native specialization check, or returns a "
                       "null value when "
                       "the specialization has no runtime precondition.",
                       "::mlir::Value", "buildCheck",
                       (ins "::mlir::OpBuilder &":$builder,
                           "::mlir::Location":$loc, "::mlir::Value":$operand)>,
       InterfaceMethod<
           "Returns the native type used to evaluate this specialization.",
           "::mlir::Type", "getTargetType", (ins)>];
}

#endif
