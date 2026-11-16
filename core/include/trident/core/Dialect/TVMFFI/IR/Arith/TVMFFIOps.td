// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPS_TD_

include "trident/core/Dialect/TVMFFI/IR/Arith/TVMFFIOpTraits.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIOpBase.td"

// Keep promotion and return inference shared across unary and binary operations.
class TVMFFI_ArithOp<string mnemonic,
                     Trait promotion = TVMFFI_ArithNumericPromotion,
                     list<Trait> traits = []>
    : TVMFFI_Op<"arith." # mnemonic,
                [InferTypeOpInterface, NoMemoryEffect, promotion] # traits> {
  let summary = "Compute boxed scalar " # mnemonic # " with numeric promotion";
  let description = [{
    Operands use TVM FFI Bool, Int or Float. Numeric arithmetic promotes Bool
    to Int and either Float operand promotes computation to Float. True division
    always computes Float. Integral operations reject Float; bitwise Bool/Bool
    retains Bool. Comparisons and logical not return Bool. Lowering extracts
    native scalars, promotes them, computes with arith and boxes the result.
    Arithmetic uses the runtime's fixed i64/f64 widths and native overflow,
    division and shift constraints. Any/Union must be explicitly narrowed.
  }];
  // Division may have undefined behavior; do not promise unconditional
  // speculation for the whole family before native arithmetic is exposed.
  let results = (outs TVMFFI_NumericType:$result);
}

class TVMFFI_BinaryArithOp<string mnemonic,
                           Trait promotion = TVMFFI_ArithNumericPromotion,
                           TypeConstraint operandType = TVMFFI_NumericType,
                           list<Trait> traits = []>
    : TVMFFI_ArithOp<mnemonic, promotion, traits> {
  let arguments = (ins operandType:$lhs, operandType:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

class TVMFFI_ComparisonOp<string mnemonic>
    : TVMFFI_BinaryArithOp<mnemonic, TVMFFI_ArithNumericPromotion,
                           TVMFFI_NumericType, [TVMFFI_ArithBoolResult]> {
  let results = (outs TVMFFI_BoolType:$result);
}

class TVMFFI_UnaryArithOp<string mnemonic, list<Trait> traits = []>
    : TVMFFI_ArithOp<mnemonic, TVMFFI_ArithNumericPromotion, traits> {
  let arguments = (ins TVMFFI_NumericType:$operand);
  let assemblyFormat = "$operand attr-dict `:` functional-type(operands, results)";
}

def TVMFFI_ArithAddOp : TVMFFI_BinaryArithOp<"add">;
def TVMFFI_ArithAndOp
    : TVMFFI_BinaryArithOp<"and", TVMFFI_ArithBitwisePromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithEqOp : TVMFFI_ComparisonOp<"eq">;
def TVMFFI_ArithFloorDivOp
    : TVMFFI_BinaryArithOp<"floordiv", TVMFFI_ArithIntegralPromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithGeOp : TVMFFI_ComparisonOp<"ge">;
def TVMFFI_ArithGtOp : TVMFFI_ComparisonOp<"gt">;
def TVMFFI_ArithLShiftOp
    : TVMFFI_BinaryArithOp<"lshift", TVMFFI_ArithIntegralPromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithLeOp : TVMFFI_ComparisonOp<"le">;
def TVMFFI_ArithLtOp : TVMFFI_ComparisonOp<"lt">;
def TVMFFI_ArithMaxOp : TVMFFI_BinaryArithOp<"max">;
def TVMFFI_ArithMinOp : TVMFFI_BinaryArithOp<"min">;
def TVMFFI_ArithModOp
    : TVMFFI_BinaryArithOp<"mod", TVMFFI_ArithIntegralPromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithMulOp : TVMFFI_BinaryArithOp<"mul">;
def TVMFFI_ArithNeOp : TVMFFI_ComparisonOp<"ne">;
def TVMFFI_ArithNegOp : TVMFFI_UnaryArithOp<"neg">;
def TVMFFI_ArithNotOp : TVMFFI_UnaryArithOp<"not", [TVMFFI_ArithBoolResult]> {
  let results = (outs TVMFFI_BoolType:$result);
}
def TVMFFI_ArithOrOp
    : TVMFFI_BinaryArithOp<"or", TVMFFI_ArithBitwisePromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithRShiftOp
    : TVMFFI_BinaryArithOp<"rshift", TVMFFI_ArithIntegralPromotion,
                           TVMFFI_IntegralType>;
def TVMFFI_ArithSubOp : TVMFFI_BinaryArithOp<"sub">;
def TVMFFI_ArithTrueDivOp : TVMFFI_BinaryArithOp<"truediv", TVMFFI_ArithFloatPromotion>;
def TVMFFI_ArithXorOp
    : TVMFFI_BinaryArithOp<"xor", TVMFFI_ArithBitwisePromotion,
                           TVMFFI_IntegralType>;

#endif
