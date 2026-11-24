// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lift-cf-to-scf --inline --symbol-dce --cse --canonicalize --verify-each | FileCheck %s

// DLPack supports inlining. Metadata reads retain memory effects, so generic
// CSE conservatively preserves the second read; scalar boxing still folds.
// CHECK-LABEL: func.func @dispatch(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !dlpack.dl_tensor)
// CHECK: %[[RANK:[a-zA-Z0-9_]+]] = dlpack.tensor.dim %[[TENSOR]]
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[RANK]], %[[EXPECTED:[a-zA-Z0-9_]+]]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[MATCH]]
// CHECK: %[[FRESH:[a-zA-Z0-9_]+]] = dlpack.tensor.dim %[[TENSOR]]
// CHECK-NOT: tvm_ffi.
// CHECK: %[[CHECKED:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[FRESH]], %[[EXPECTED]]
// CHECK: scf.if %[[CHECKED]]
// CHECK-NOT: func.call @checked
// CHECK: func.call @run
// CHECK: } else {
// CHECK: func.call @reject
// CHECK: } else {
// CHECK: func.call @reject
// CHECK: return %[[RESULT]]
func.func @dispatch(%tensor: !dlpack.dl_tensor) -> i64 {
  %rank = dlpack.tensor.dim %tensor : !dlpack.dl_tensor
  %one = arith.constant 1 : i64
  %matches = arith.cmpi eq, %rank, %one : i64
  %result = scf.if %matches -> i64 {
    %value = func.call @checked(%tensor) : (!dlpack.dl_tensor) -> i64
    scf.yield %value : i64
  } else {
    %value = func.call @reject() : () -> i64
    scf.yield %value : i64
  }
  return %result : i64
}

func.func private @checked(%tensor: !dlpack.dl_tensor) -> i64 {
  %rank = dlpack.tensor.dim %tensor : !dlpack.dl_tensor
  %boxed = tvm_ffi_abi.to %rank : i64 -> !tvm_ffi_abi.int
  %wide = tvm_ffi_api.upcast %boxed : !tvm_ffi_abi.int -> !tvm_ffi_api.any
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_api.any -> !tvm_ffi_abi.int
  %native = tvm_ffi_api.get %narrow : !tvm_ffi_abi.int -> i64
  %one = arith.constant 1 : i64
  %matches = arith.cmpi eq, %native, %one : i64
  cf.cond_br %matches, ^accepted, ^rejected
^accepted:
  %value = func.call @run() : () -> i64
  return %value : i64
^rejected:
  %error = func.call @reject() : () -> i64
  return %error : i64
}

func.func private @reject() -> i64
func.func private @run() -> i64
