// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s

// Cast chains borrow their source; exports retain the semantic source once per
// occurrence without adding local cleanup responsibility for each cast.
// CHECK-LABEL: func.func @cast_exports(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[OWNED:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[OWNED]]
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK: %[[EXPORT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NARROW]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[EXPORT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[EXPORT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OWNED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[EXPORT]], %[[EXPORT]]
func.func @cast_exports(%input: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.any) {
  %owned = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %wide = tvm_ffi.upcast %owned : !tvm_ffi.tensor -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.tensor
  %export = tvm_ffi.upcast %narrow : !tvm_ffi.tensor -> !tvm_ffi.any
  return %export, %export : !tvm_ffi.any, !tvm_ffi.any
}

// Every Opaque export is retained, including casts from Base references and POD.
// The casts borrow their inputs and register no local cleanup.
// CHECK-LABEL: func.func @opaque_exports(
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[POD:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[BASE]]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[POD]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OBJECT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SCALAR]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[OBJECT]], %[[SCALAR]]
func.func @opaque_exports(%base: !tvm_ffi.c.tensor, %pod: !tvm_ffi.int) -> (!tvm_ffi.any, !tvm_ffi.any) {
  %object = tvm_ffi.upcast %base : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %scalar = tvm_ffi.upcast %pod : !tvm_ffi.int -> !tvm_ffi.any
  return %object, %scalar : !tvm_ffi.any, !tvm_ffi.any
}

// Array and tuple extraction must use the same expanded string representation.
// CHECK-LABEL: func.func @string_elements(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]] : i64 -> !tvm_ffi.int
// CHECK: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ARRAY_HANDLE]] : !tvm_ffi.c.function(%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY_HANDLE]] : !tvm_ffi.c.function
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO]] : i64 -> !tvm_ffi.int
// CHECK: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[TUPLE_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[OFFSET]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
func.func @string_elements(%array: !tvm_ffi.array<!tvm_ffi.str>, %tuple: !tvm_ffi.tuple<!tvm_ffi.str>, %index: i64) -> (!tvm_ffi.str, !tvm_ffi.str) {
  %first = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.str> -> !tvm_ffi.str
  %second = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.str>) -> !tvm_ffi.str
  return %first, %second : !tvm_ffi.str, !tvm_ffi.str
}

// Even unions whose members share a representation lower to CAny.
// CHECK-LABEL: func.func @union_collapse(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK: return %[[VALUE]] : !tvm_ffi.c.any
func.func @union_collapse(%value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor>) -> !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor> {
  return %value : !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor>
}

// Union members are erased from the representation.
// CHECK-LABEL: func.func @union_order(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// CHECK: return %[[VALUE]] : !tvm_ffi.c.any
func.func @union_order(%value: !tvm_ffi.union<!tvm_ffi.c.str, !tvm_ffi.str, !tvm_ffi.small_str, !tvm_ffi.tensor, !tvm_ffi.c.tensor>) -> !tvm_ffi.union<!tvm_ffi.c.str, !tvm_ffi.str, !tvm_ffi.small_str, !tvm_ffi.tensor, !tvm_ffi.c.tensor> {
  return %value : !tvm_ffi.union<!tvm_ffi.c.str, !tvm_ffi.str, !tvm_ffi.small_str, !tvm_ffi.tensor, !tvm_ffi.c.tensor>
}
