// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @tuple(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[FIRST]], %[[SECOND]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[TUPLE]]
// CHECK: tvm_ffi.return %[[RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>>
tvm_ffi.func @tuple(%first: !tvm_ffi.int, %second: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>> {
  %tuple = tvm_ffi.tuple.construct(%first, %second) : (!tvm_ffi.int, !tvm_ffi.tensor) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>
  %result = tvm_ffi.monad.pure %tuple : (!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>>
}

// CHECK-LABEL: tvm_ffi.func @unit(
// CHECK-SAME: %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
// CHECK: tvm_ffi.return %[[UNIT]] : !tvm_ffi.monad.result<!tvm_ffi.none>
tvm_ffi.func @unit(%unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  tvm_ffi.return %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
}
