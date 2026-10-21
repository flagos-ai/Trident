// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'

// The shared Array/Tuple construction protocol borrows its elements. Temporary
// runtime handles are released immediately; containers are released in reverse
// construction order after all exports. Semantic strings keep their tagged
// cleanup contract even when the underlying constant uses static storage.
// CHECK-LABEL: func.func @nested_construction()
// CHECK-NEXT: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "text"
// CHECK-NEXT: %[[TEXT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW]]
// CHECK-NEXT: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ARRAY_HANDLE]] : !tvm_ffi.c.function(%[[TEXT]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[TUPLE_HANDLE]] : !tvm_ffi.c.function(%[[ARRAY]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[TUPLE]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[TEXT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TEXT]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[TUPLE]], %[[TEXT]]
func.func @nested_construction() -> (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>, !tvm_ffi.str) {
  %text = tvm_ffi.constant.str "text"
  %array = tvm_ffi.array.create(%text, %text) : (!tvm_ffi.str, !tvm_ffi.str) -> !tvm_ffi.array<!tvm_ffi.str>
  %tuple = tvm_ffi.tuple.construct(%array, %text) : (!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str) -> !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>
  return %tuple, %text : !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>, !tvm_ffi.str
}
