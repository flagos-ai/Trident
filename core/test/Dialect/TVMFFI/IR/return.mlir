// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @payloads(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<i64>, %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<i32>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>)
// CHECK: tvm_ffi.return %[[FIRST]], %[[UNIT]], %[[SECOND]] : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
tvm_ffi.func @payloads(%first: !tvm_ffi.monad.result<i64>, %unit: !tvm_ffi.monad.result<!tvm_ffi.none>, %second: !tvm_ffi.monad.result<i32>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>) {

  tvm_ffi.return %first, %unit, %second : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
}

// CHECK-LABEL: tvm_ffi.func @unit(
// CHECK-SAME: %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
// CHECK: tvm_ffi.return %[[UNIT]] : !tvm_ffi.monad.result<!tvm_ffi.none>
tvm_ffi.func @unit(%unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  tvm_ffi.return %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
}
