// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-each | trident-core-opt --split-input-file --verify-each | FileCheck %s

// CHECK-LABEL: func.func @captured(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: tvm_ffi_abi.defer {
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @captured(%input: !tvm_ffi_abi.tensor) {
  tvm_ffi_abi.defer {
    tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer_yield
  }
  return
}

// Even an empty defer is conservatively effectful until explicitly lowered.
// CHECK-LABEL: func.func @empty()
// CHECK: tvm_ffi_abi.defer {
// CHECK-NEXT: tvm_ffi_abi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @empty() {
  tvm_ffi_abi.defer {
    tvm_ffi_abi.defer_yield
  }
  return
}

// -----



// -----

// CHECK-LABEL: func.func @to_native_scalars(
// CHECK-SAME: [[BOOL_ARG:%[a-zA-Z0-9_]+]]: i1, [[INT_ARG:%[a-zA-Z0-9_]+]]: i64, [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: f64)
// CHECK: [[BOOL:%[a-zA-Z0-9_]+]] = tvm_ffi_abi.to [[BOOL_ARG]] : i1 -> !tvm_ffi_abi.bool
// CHECK: [[INT:%[a-zA-Z0-9_]+]] = tvm_ffi_abi.to [[INT_ARG]] : i64 -> !tvm_ffi_abi.int
// CHECK: [[FLOAT:%[a-zA-Z0-9_]+]] = tvm_ffi_abi.to [[FLOAT_ARG]] : f64 -> !tvm_ffi_abi.float
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.int, !tvm_ffi_abi.float) {
  %bool_value = tvm_ffi_abi.to %bool : i1 -> !tvm_ffi_abi.bool
  %int_value = tvm_ffi_abi.to %int : i64 -> !tvm_ffi_abi.int
  %float_value = tvm_ffi_abi.to %float : f64 -> !tvm_ffi_abi.float
  return %bool_value, %int_value, %float_value
     : !tvm_ffi_abi.bool, !tvm_ffi_abi.int, !tvm_ffi_abi.float
}
