// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -lower-tvm-ffi-monad -canonicalize | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @failed_any()
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK-NOT: tvm_ffi.monad.unpack
// CHECK: builtin.unrealized_conversion_cast %[[FALSE]], %[[TRUE:[a-zA-Z0-9_]+]]
// CHECK: builtin.unrealized_conversion_cast %[[ANY]], %[[TRUE]]
tvm_ffi.func @failed_any() -> (!tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<!tvm_ffi.any>) {
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  %success, %value = tvm_ffi.monad.unpack %failed : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  %returned:2 = tvm_ffi.monad.pure %success, %value : (i1, !tvm_ffi.any) -> (!tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<!tvm_ffi.any>)
  tvm_ffi.return %returned#0, %returned#1 : !tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<!tvm_ffi.any>
}

// CHECK-LABEL: tvm_ffi.func @native_success(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK-NOT: tvm_ffi.to
// CHECK-NOT: tvm_ffi.upcast
// CHECK: builtin.unrealized_conversion_cast %[[INPUT]], %[[TRUE]]
tvm_ffi.func @native_success(%input: i64) -> !tvm_ffi.monad.result<i64> {
  %pure = tvm_ffi.monad.pure %input : (i64) -> !tvm_ffi.monad.result<i64>
  %success, %value = tvm_ffi.monad.unpack %pure : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  %returned = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<i64>
}
