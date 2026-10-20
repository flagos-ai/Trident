// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s

// Decomposition propagates explicit cleanup for independent references; temporary handles are explicitly
// released and must not acquire another release on the second cleanup pass.
// CHECK-LABEL: func.func @array(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[CREATE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE_HANDLE]](%[[INPUT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CREATE_HANDLE]]
// CHECK: %[[GET_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET_HANDLE]](%[[ARRAY]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[GET_HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY]]
// CHECK-NEXT: return
func.func @array(%input: !tvm_ffi.tensor) {
  %array = tvm_ffi.array.create(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.array<!tvm_ffi.tensor>
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %array[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.tensor
  return {tvm_ffi.retain}
}

// Each unconstructed result inherits the operation cleanup marker.
// CHECK-LABEL: func.func @tuple(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FIRST_HANDLE]](%[[INPUT]], %[[ZERO:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST_HANDLE]]
// CHECK: %[[SECOND_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SECOND_HANDLE]](%[[INPUT]], %[[ONE:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND_HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-NEXT: return
func.func @tuple(%input: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) {
  %elements:2 = tvm_ffi.tuple.unconstruct %input {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) -> (!tvm_ffi.tensor, !tvm_ffi.tensor)
  return {tvm_ffi.retain}
}

// Scalar extraction records intent too; cleanup releases only its temporary handle.
// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]]
// CHECK-NEXT: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.int
func.func @scalar(%array: !tvm_ffi.array<!tvm_ffi.int>, %index: i64) -> !tvm_ffi.int {
  %value = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return {tvm_ffi.retain} %value : !tvm_ffi.int
}
