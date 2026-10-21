// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s --implicit-check-not="tvm_ffi.upcast" --implicit-check-not="tvm_ffi.downcast"

// Tagged CAny, native function pointers and indexed POD identity conversions.
// Semantic type normalization and folding belong to the dialect fixtures.

// CHECK-LABEL: func.func @any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @any(%arg: !tvm_ffi.c.any) -> !tvm_ffi.c.any {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.c.any -> !tvm_ffi.c.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.c.any -> !tvm_ffi.c.any
  return %narrow : !tvm_ffi.c.any
}

// CHECK-LABEL: func.func @function(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @function(%arg: !tvm_ffi.c.function) -> !tvm_ffi.c.function {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.c.function -> !tvm_ffi.c.function
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.c.function -> !tvm_ffi.c.function
  return %narrow : !tvm_ffi.c.function
}

// CHECK-LABEL: func.func @int(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @int(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.int
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.int -> !tvm_ffi.int
  return %narrow : !tvm_ffi.int
}
