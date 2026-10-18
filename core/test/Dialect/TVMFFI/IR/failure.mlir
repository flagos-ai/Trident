// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @failure(
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]]:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: tvm_ffi.return %[[FAILED]]#0, %[[FAILED]]#1, %[[FAILED]]#2
tvm_ffi.func @failure() -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %failed:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>

  tvm_ffi.return %failed#0, %failed#1, %failed#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>
}
