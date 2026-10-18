// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: func.func @array_ops(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.array.create(%[[VALUE]]) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
// CHECK: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.array.get %[[ARRAY]][%[[INDEX]]] : <!tvm_ffi.int> -> !tvm_ffi.int
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.array.size %[[ARRAY]] : <!tvm_ffi.int> -> !tvm_ffi.int
// CHECK: return %[[ELEMENT]], %[[SIZE]]
func.func @array_ops(%value: !tvm_ffi.int, %index: i64) -> (!tvm_ffi.int, !tvm_ffi.int) {
  %array = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
  %element = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  %size = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %element, %size : !tvm_ffi.int, !tvm_ffi.int
}

// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
// CHECK: tvm_ffi.tuple.unconstruct %[[EMPTY]] : (!tvm_ffi.tuple<>) -> ()
// CHECK: return %[[EMPTY]] : !tvm_ffi.tuple<>
func.func @empty_tuple() -> !tvm_ffi.tuple<> {
  %tuple = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<>) -> ()
  return %tuple : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @pack_unpack(
// CHECK-SAME: %[[INTEGER:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[BOOLEAN:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array<!tvm_ffi.int>)
// CHECK: %[[SINGLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[INTEGER]]) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[SINGLE]], %[[BOOLEAN]], %[[ARRAY]])
// CHECK: %[[ITEMS:[a-zA-Z0-9_]+]]:3 = tvm_ffi.tuple.unconstruct %[[TUPLE]]
// CHECK: return %[[ITEMS]]#0, %[[ITEMS]]#1, %[[ITEMS]]#2
func.func @pack_unpack(%integer: !tvm_ffi.int, %boolean: !tvm_ffi.bool, %array: !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>) {
  %single = tvm_ffi.tuple.construct(%integer) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %tuple = tvm_ffi.tuple.construct(%single, %boolean, %array) : (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>>
  %nested, %bool, %list = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>>) -> (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>)
  return %nested, %bool, %list : !tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>
}
