// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPTRAITS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPTRAITS_TD_

include "mlir/IR/OpBase.td"

class TVMFFI_ArithTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::trident::tvm_ffi::OpTrait";
}

def TVMFFI_ArithBitwisePromotion : TVMFFI_ArithTrait<"ArithBitwisePromotion">;
def TVMFFI_ArithBoolResult : TVMFFI_ArithTrait<"ArithBoolResult">;
def TVMFFI_ArithFloatPromotion : TVMFFI_ArithTrait<"ArithFloatPromotion">;
def TVMFFI_ArithIntegralPromotion : TVMFFI_ArithTrait<"ArithIntegralPromotion">;
def TVMFFI_ArithNumericPromotion : TVMFFI_ArithTrait<"ArithNumericPromotion">;

#endif
