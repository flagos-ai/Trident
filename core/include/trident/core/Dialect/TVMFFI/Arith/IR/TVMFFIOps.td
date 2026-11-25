// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_ARITH_IR_ARITH_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_ARITH_IR_ARITH_TVMFFIOPS_TD_

include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIOpTraits.td"
include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIOpBase.td"

// Keep promotion and return inference shared across unary and binary
// operations.
class TVMFFIArith_ScalarOp<string mnemonic,
                           Trait promotion = TVMFFIArith_NumericPromotion,
                           list<Trait> traits = []>
    : TVMFFIArith_Op<mnemonic, [InferTypeOpInterface, NoMemoryEffect,
                                promotion]#traits> {
  let summary = "Compute boxed scalar "#mnemonic#" with numeric promotion";
  // clang-format off
  let description = [{
    Operands use TVM FFI Bool, Int or Float. Numeric arithmetic promotes Bool
    to Int and either Float operand promotes computation to Float. True division
    always computes Float. Integral operations reject Float; bitwise Bool/Bool
    retains Bool. Comparisons and logical not return Bool. Lowering extracts
    native scalars, promotes them, computes with arith and boxes the result.
    Arithmetic uses the runtime's fixed i64/f64 widths and native overflow,
    division and shift constraints. Any/Union must be explicitly narrowed.
  }];
  // clang-format on
  // Division may have undefined behavior; do not promise unconditional
  // speculation for the whole family before native arithmetic is exposed.
  let results = (outs TVMFFIABI_NumericType:$result);
}

class TVMFFIArith_BinaryArithOp<
    string mnemonic, Trait promotion = TVMFFIArith_NumericPromotion,
    TypeConstraint operandType = TVMFFIABI_NumericType, list<Trait> traits = []>
    : TVMFFIArith_ScalarOp<mnemonic, promotion, traits> {
  let arguments = (ins operandType:$lhs, operandType:$rhs);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class TVMFFIArith_ComparisonOp<string mnemonic>
    : TVMFFIArith_BinaryArithOp<
          mnemonic, TVMFFIArith_NumericPromotion,
          TVMFFIABI_NumericType, [TVMFFIArith_BoolResult]> {
  let results = (outs TVMFFIABI_BoolType:$result);
}

class TVMFFIArith_UnaryArithOp<string mnemonic, list<Trait> traits = []>
    : TVMFFIArith_ScalarOp<mnemonic, TVMFFIArith_NumericPromotion, traits> {
  let arguments = (ins TVMFFIABI_NumericType:$operand);
  let assemblyFormat =
      "$operand attr-dict `:` functional-type(operands, results)";
}

def TVMFFIArith_AddOp : TVMFFIArith_BinaryArithOp<"add">;
def TVMFFIArith_AndOp
    : TVMFFIArith_BinaryArithOp<"and", TVMFFIArith_BitwisePromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_EqOp : TVMFFIArith_ComparisonOp<"eq">;
def TVMFFIArith_FloorDivOp
    : TVMFFIArith_BinaryArithOp<"floordiv", TVMFFIArith_IntegralPromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_GeOp : TVMFFIArith_ComparisonOp<"ge">;
def TVMFFIArith_GtOp : TVMFFIArith_ComparisonOp<"gt">;
def TVMFFIArith_LShiftOp
    : TVMFFIArith_BinaryArithOp<"lshift", TVMFFIArith_IntegralPromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_LeOp : TVMFFIArith_ComparisonOp<"le">;
def TVMFFIArith_LtOp : TVMFFIArith_ComparisonOp<"lt">;
def TVMFFIArith_MaxOp : TVMFFIArith_BinaryArithOp<"max">;
def TVMFFIArith_MinOp : TVMFFIArith_BinaryArithOp<"min">;
def TVMFFIArith_ModOp
    : TVMFFIArith_BinaryArithOp<"mod", TVMFFIArith_IntegralPromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_MulOp : TVMFFIArith_BinaryArithOp<"mul">;
def TVMFFIArith_NeOp : TVMFFIArith_ComparisonOp<"ne">;
def TVMFFIArith_NegOp : TVMFFIArith_UnaryArithOp<"neg">;
def TVMFFIArith_NotOp
    : TVMFFIArith_UnaryArithOp<"not", [TVMFFIArith_BoolResult]> {
  let results = (outs TVMFFIABI_BoolType:$result);
}
def TVMFFIArith_OrOp
    : TVMFFIArith_BinaryArithOp<"or", TVMFFIArith_BitwisePromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_RShiftOp
    : TVMFFIArith_BinaryArithOp<"rshift", TVMFFIArith_IntegralPromotion,
                                TVMFFIABI_IntegralType>;
def TVMFFIArith_SubOp : TVMFFIArith_BinaryArithOp<"sub">;
def TVMFFIArith_TrueDivOp
    : TVMFFIArith_BinaryArithOp<"truediv", TVMFFIArith_FloatPromotion>;
def TVMFFIArith_XorOp
    : TVMFFIArith_BinaryArithOp<"xor", TVMFFIArith_BitwisePromotion,
                                TVMFFIABI_IntegralType>;

#endif
