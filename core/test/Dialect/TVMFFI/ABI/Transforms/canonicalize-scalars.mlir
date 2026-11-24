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
  %yes = tvm_ffi_abi.constant.bool true
  %no = tvm_ffi_abi.constant.bool false
  %int = tvm_ffi_abi.constant.int 42
  %float = tvm_ffi_abi.constant.float 2.5
  %a = tvm_ffi_abi.get %yes : !tvm_ffi_abi.bool -> i1
  %b = tvm_ffi_abi.get %no : !tvm_ffi_abi.bool -> i1
  %c = tvm_ffi_abi.get %int : !tvm_ffi_abi.int -> i64
  %d = tvm_ffi_abi.get %float : !tvm_ffi_abi.float -> f64
  return %a, %b, %c, %d : i1, i1, i64, f64
}

// CHECK-LABEL: func.func @roundtrip(
// CHECK-SAME: %[[NATIVE:[a-zA-Z0-9_]+]]: i1, %[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: return %[[NATIVE]], %[[BOXED]]
func.func @roundtrip(%native: i1, %boxed: !tvm_ffi_abi.int) -> (i1, !tvm_ffi_abi.int) {
  %a = tvm_ffi_abi.to %native : i1 -> !tvm_ffi_abi.bool
  %b = tvm_ffi_abi.get %a : !tvm_ffi_abi.bool -> i1
  %c = tvm_ffi_abi.get %boxed : !tvm_ffi_abi.int -> i64
  %d = tvm_ffi_abi.to %c : i64 -> !tvm_ffi_abi.int
  return %b, %d : i1, !tvm_ffi_abi.int
}

// A native block argument has no defining GetOp; boxing must survive.
// CHECK-LABEL: func.func @box_argument(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64)
// CHECK-NEXT: %[[BOX:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[VALUE]] : i64 -> !tvm_ffi_abi.int
// CHECK-NEXT: return %[[BOX]]
func.func @box_argument(%value: i64) -> !tvm_ffi_abi.int {
  %boxed = tvm_ffi_abi.to %value : i64 -> !tvm_ffi_abi.int
  return %boxed : !tvm_ffi_abi.int
}
