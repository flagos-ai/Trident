// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --convert-tvm-ffi-to-llvm --reconcile-unrealized-casts | FileCheck %s

// CHECK-LABEL: func.func @any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @any(%arg: !tvm_ffi.any) -> !tvm_ffi.any {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.any -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.any
  return %narrow : !tvm_ffi.any
}

// CHECK-LABEL: func.func @array(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @array(%arg: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.array<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.array<!tvm_ffi.int>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.array<!tvm_ffi.int>
  return %narrow : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @function(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @function(%arg: !tvm_ffi.c.function) -> !tvm_ffi.c.function {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.c.function -> !tvm_ffi.c.function
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.c.function -> !tvm_ffi.c.function
  return %narrow : !tvm_ffi.c.function
}

// CHECK-LABEL: func.func @int(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @int(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.int
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.int -> !tvm_ffi.int
  return %narrow : !tvm_ffi.int
}

// CHECK-LABEL: func.func @tuple(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @tuple(%arg: !tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.tuple<!tvm_ffi.int> -> !tvm_ffi.tuple<!tvm_ffi.int>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.tuple<!tvm_ffi.int> -> !tvm_ffi.tuple<!tvm_ffi.int>
  return %narrow : !tvm_ffi.tuple<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK: return %[[VALUE]]
func.func @union(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %narrow : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

