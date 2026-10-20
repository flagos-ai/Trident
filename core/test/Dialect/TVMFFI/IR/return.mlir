// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @tuple(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[FIRST]], %[[SECOND]])
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TUPLE]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: tvm_ffi.return %[[OK]], %[[PAYLOAD]] : !tvm_ffi.bool, !tvm_ffi.any
tvm_ffi.func @tuple(%first: !tvm_ffi.int, %second: !tvm_ffi.tensor) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor> {
  %tuple = tvm_ffi.tuple.construct(%first, %second) : (!tvm_ffi.int, !tvm_ffi.tensor) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor>
  %payload = tvm_ffi.upcast %tuple : !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.tensor> -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// CHECK-LABEL: tvm_ffi.func @failure(
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool false
// CHECK: tvm_ffi.return %[[OK]], %[[PAYLOAD]]
tvm_ffi.func @failure() -> !tvm_ffi.tensor {
  %none = tvm_ffi.constant.none
  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool false
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}
