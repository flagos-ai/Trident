// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s

// Constant Monad success flags must become native conditions, so SCF can fold.
// CHECK-LABEL: func.func @constants(
// CHECK-DAG: %[[YES:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK-DAG: %[[NO:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK-DAG: %[[INT:[a-zA-Z0-9_]+]] = arith.constant 42 : i64
// CHECK-DAG: %[[FLOAT:[a-zA-Z0-9_]+]] = arith.constant 2.500000e+00 : f64
// CHECK-NOT: tvm_ffi.
// CHECK: return %[[YES]], %[[NO]], %[[INT]], %[[FLOAT]]
func.func @constants() -> (i1, i1, i64, f64) {
  %yes = tvm_ffi.constant.bool true
  %no = tvm_ffi.constant.bool false
  %int = tvm_ffi.constant.int 42
  %float = tvm_ffi.constant.float 2.5
  %a = tvm_ffi.get %yes : !tvm_ffi.bool -> i1
  %b = tvm_ffi.get %no : !tvm_ffi.bool -> i1
  %c = tvm_ffi.get %int : !tvm_ffi.int -> i64
  %d = tvm_ffi.get %float : !tvm_ffi.float -> f64
  return %a, %b, %c, %d : i1, i1, i64, f64
}

// CHECK-LABEL: func.func @roundtrip(
// CHECK-SAME: %[[NATIVE:[a-zA-Z0-9_]+]]: i1, %[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: return %[[NATIVE]], %[[BOXED]]
func.func @roundtrip(%native: i1, %boxed: !tvm_ffi.int) -> (i1, !tvm_ffi.int) {
  %a = tvm_ffi.to %native : i1 -> !tvm_ffi.bool
  %b = tvm_ffi.get %a : !tvm_ffi.bool -> i1
  %c = tvm_ffi.get %boxed : !tvm_ffi.int -> i64
  %d = tvm_ffi.to %c : i64 -> !tvm_ffi.int
  return %b, %d : i1, !tvm_ffi.int
}

// A native block argument has no defining GetOp; boxing must survive.
// CHECK-LABEL: func.func @box_argument(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64)
// CHECK-NEXT: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[VALUE]] : i64 -> !tvm_ffi.int
// CHECK-NEXT: return %[[BOX]]
func.func @box_argument(%value: i64) -> !tvm_ffi.int {
  %boxed = tvm_ffi.to %value : i64 -> !tvm_ffi.int
  return %boxed : !tvm_ffi.int
}
