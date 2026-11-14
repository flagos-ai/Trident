// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --loop-invariant-code-motion --lower-tvm-ffi-arith --verify-each | FileCheck %s

// Each case isolates a distinct promotion or native computation rule.
// CHECK-LABEL: func.func @bool_add(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.bool)
// CHECK: %[[X:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.bool -> i1
// CHECK: %[[L:[a-zA-Z0-9_]+]] = arith.extui %[[X]] : i1 to i64
// CHECK: %[[Y:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.bool -> i1
// CHECK: %[[R:[a-zA-Z0-9_]+]] = arith.extui %[[Y]] : i1 to i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[L]], %[[R]] : i64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[SUM]] : i64 -> !tvm_ffi.int
// CHECK: return %[[BOX]]
func.func @bool_add(%a: !tvm_ffi.bool, %b: !tvm_ffi.bool) -> !tvm_ffi.int {
  %result = tvm_ffi.arith.add %a, %b : (!tvm_ffi.bool, !tvm_ffi.bool) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @bool_bitwise(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.bool)
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.bool -> i1
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.bool -> i1
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = arith.xori %[[L]], %[[R]] : i1
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[VALUE]] : i1 -> !tvm_ffi.bool
// CHECK: return %[[BOX]]
func.func @bool_bitwise(%a: !tvm_ffi.bool, %b: !tvm_ffi.bool) -> !tvm_ffi.bool {
  %result = tvm_ffi.arith.xor %a, %b : (!tvm_ffi.bool, !tvm_ffi.bool) -> !tvm_ffi.bool
  return %result : !tvm_ffi.bool
}

// A zero-trip loop must not evaluate a potentially undefined division.
// CHECK-LABEL: func.func @conditional_division(
// CHECK-SAME: %[[LOWER:[a-zA-Z0-9_]+]]: index, %[[UPPER:[a-zA-Z0-9_]+]]: index, %[[STEP:[a-zA-Z0-9_]+]]: index, %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.for %[[INDEX:[a-zA-Z0-9_]+]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] iter_args(%[[CURRENT:[a-zA-Z0-9_]+]] = %[[ZERO]]) -> (!tvm_ffi.int) {
// CHECK: %[[L:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.int -> i64
// CHECK: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.int -> i64
// CHECK: %[[QUOTIENT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[L]], %[[R]] : i64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[QUOTIENT]] : i64 -> !tvm_ffi.int
// CHECK: scf.yield %[[BOX]]
// CHECK: return %[[RESULT]]
func.func @conditional_division(%lower: index, %upper: index, %step: index, %lhs: !tvm_ffi.int, %rhs: !tvm_ffi.int) -> !tvm_ffi.int {
  %zero = tvm_ffi.constant.int 0
  %result = scf.for %index = %lower to %upper step %step iter_args(%current = %zero) -> !tvm_ffi.int {
    %quotient = tvm_ffi.arith.floordiv %lhs, %rhs : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.int
    scf.yield %quotient : !tvm_ffi.int
  }
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @mixed_float(
// CHECK-SAME: %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[I:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[F:[a-zA-Z0-9_]+]]: !tvm_ffi.float)
// CHECK-DAG: %[[BV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.bool -> i1
// CHECK-DAG: %[[BF:[a-zA-Z0-9_]+]] = arith.uitofp %[[BV]] : i1 to f64
// CHECK-DAG: %[[FV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[F]] : !tvm_ffi.float -> f64
// CHECK-DAG: %[[SUM:[a-zA-Z0-9_]+]] = arith.addf %[[BF]], %[[FV]] : f64
// CHECK-DAG: %[[IV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[I]] : !tvm_ffi.int -> i64
// CHECK-DAG: %[[IF:[a-zA-Z0-9_]+]] = arith.sitofp %[[IV]] : i64 to f64
// CHECK-DAG: %[[PRODUCT:[a-zA-Z0-9_]+]] = arith.mulf %[[SUM]], %[[IF]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PRODUCT]] : f64 -> !tvm_ffi.float
// CHECK: return %[[BOX]]
func.func @mixed_float(%b: !tvm_ffi.bool, %i: !tvm_ffi.int, %f: !tvm_ffi.float) -> !tvm_ffi.float {
  %sum = tvm_ffi.arith.add %b, %f : (!tvm_ffi.bool, !tvm_ffi.float) -> !tvm_ffi.float
  %result = tvm_ffi.arith.mul %sum, %i : (!tvm_ffi.float, !tvm_ffi.int) -> !tvm_ffi.float
  return %result : !tvm_ffi.float
}

// CHECK-LABEL: func.func @negative_zero(
// CHECK-SAME: %[[F:[a-zA-Z0-9_]+]]: !tvm_ffi.float)
// CHECK: %[[V:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[F]] : !tvm_ffi.float -> f64
// CHECK: %[[NEG:[a-zA-Z0-9_]+]] = arith.negf %[[V]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[NEG]] : f64 -> !tvm_ffi.float
// CHECK: return %[[BOX]]
func.func @negative_zero(%f: !tvm_ffi.float) -> !tvm_ffi.float {
  %result = tvm_ffi.arith.neg %f : (!tvm_ffi.float) -> !tvm_ffi.float
  return %result : !tvm_ffi.float
}

// CHECK-LABEL: func.func @numeric_comparison(
// CHECK-SAME: %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[I:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-DAG: %[[BV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.bool -> i1
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = arith.extui %[[BV]] : i1 to i64
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[I]] : !tvm_ffi.int -> i64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = arith.cmpi slt, %[[L]], %[[R]] : i64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PRED]] : i1 -> !tvm_ffi.bool
// CHECK: return %[[BOX]]
func.func @numeric_comparison(%b: !tvm_ffi.bool, %i: !tvm_ffi.int) -> !tvm_ffi.bool {
  %result = tvm_ffi.arith.lt %b, %i : (!tvm_ffi.bool, !tvm_ffi.int) -> !tvm_ffi.bool
  return %result : !tvm_ffi.bool
}

// CHECK-LABEL: func.func @numeric_not(
// CHECK-SAME: %[[F:[a-zA-Z0-9_]+]]: !tvm_ffi.float)
// CHECK-DAG: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG: %[[V:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[F]] : !tvm_ffi.float -> f64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = arith.cmpf oeq, %[[V]], %[[ZERO]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PRED]] : i1 -> !tvm_ffi.bool
// CHECK: return %[[BOX]]
func.func @numeric_not(%f: !tvm_ffi.float) -> !tvm_ffi.bool {
  %result = tvm_ffi.arith.not %f : (!tvm_ffi.float) -> !tvm_ffi.bool
  return %result : !tvm_ffi.bool
}

// CHECK-LABEL: func.func @ordered_comparison(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.float, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.float)
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.float -> f64
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.float -> f64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = arith.cmpf oge, %[[L]], %[[R]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PRED]] : i1 -> !tvm_ffi.bool
// CHECK: return %[[BOX]]
func.func @ordered_comparison(%a: !tvm_ffi.float, %b: !tvm_ffi.float) -> !tvm_ffi.bool {
  %result = tvm_ffi.arith.ge %a, %b : (!tvm_ffi.float, !tvm_ffi.float) -> !tvm_ffi.bool
  return %result : !tvm_ffi.bool
}

// CHECK-LABEL: func.func @remainder(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.int -> i64
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.int -> i64
// CHECK: %[[QUOTIENT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[L]], %[[R]] : i64
// CHECK: %[[PRODUCT:[a-zA-Z0-9_]+]] = arith.muli %[[QUOTIENT]], %[[R]] : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = arith.subi %[[L]], %[[PRODUCT]] : i64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[VALUE]] : i64 -> !tvm_ffi.int
// CHECK: return %[[BOX]]
func.func @remainder(%a: !tvm_ffi.int, %b: !tvm_ffi.int) -> !tvm_ffi.int {
  %result = tvm_ffi.arith.mod %a, %b : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @true_division(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.bool)
// CHECK-DAG: %[[AV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.int -> i64
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = arith.sitofp %[[AV]] : i64 to f64
// CHECK-DAG: %[[BV:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.bool -> i1
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = arith.uitofp %[[BV]] : i1 to f64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = arith.divf %[[L]], %[[R]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[VALUE]] : f64 -> !tvm_ffi.float
// CHECK: return %[[BOX]]
func.func @true_division(%a: !tvm_ffi.int, %b: !tvm_ffi.bool) -> !tvm_ffi.float {
  %result = tvm_ffi.arith.truediv %a, %b : (!tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.float
  return %result : !tvm_ffi.float
}

// CHECK-LABEL: func.func @unordered_comparison(
// CHECK-SAME: %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.float, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.float)
// CHECK-DAG: %[[L:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[A]] : !tvm_ffi.float -> f64
// CHECK-DAG: %[[R:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[B]] : !tvm_ffi.float -> f64
// CHECK: %[[PRED:[a-zA-Z0-9_]+]] = arith.cmpf une, %[[L]], %[[R]] : f64
// CHECK: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PRED]] : i1 -> !tvm_ffi.bool
// CHECK: return %[[BOX]]
func.func @unordered_comparison(%a: !tvm_ffi.float, %b: !tvm_ffi.float) -> !tvm_ffi.bool {
  %result = tvm_ffi.arith.ne %a, %b : (!tvm_ffi.float, !tvm_ffi.float) -> !tvm_ffi.bool
  return %result : !tvm_ffi.bool
}
