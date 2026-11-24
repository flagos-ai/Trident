// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_ARITH_IR_ARITH_TVMFFIOPTRAITS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_ARITH_IR_ARITH_TVMFFIOPTRAITS_TD_

include "mlir/IR/OpBase.td"

class TVMFFIArith_Trait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::trident::tvm_ffi::arith::OpTrait";
}

def TVMFFIArith_BitwisePromotion : TVMFFIArith_Trait<"ArithBitwisePromotion">;
def TVMFFIArith_BoolResult : TVMFFIArith_Trait<"ArithBoolResult">;
def TVMFFIArith_FloatPromotion : TVMFFIArith_Trait<"ArithFloatPromotion">;
def TVMFFIArith_IntegralPromotion : TVMFFIArith_Trait<"ArithIntegralPromotion">;
def TVMFFIArith_NumericPromotion : TVMFFIArith_Trait<"ArithNumericPromotion">;

#endif
