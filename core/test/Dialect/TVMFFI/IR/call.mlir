// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | FileCheck %s

tvm_ffi.func @callee(!tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi}

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[MONAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.call @callee(%[[INPUT]]) {tvm_ffi.cleanup}
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @callee(%[[INPUT]]) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: return %[[MONAD]], %[[OK]], %[[VALUE]]
func.func @invoke(%input: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.bool, !tvm_ffi.any) {
  %monad = tvm_ffi.monad.call @callee(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  %ok, %value = tvm_ffi.call @callee(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %monad, %ok, %value : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.bool, !tvm_ffi.any
}
