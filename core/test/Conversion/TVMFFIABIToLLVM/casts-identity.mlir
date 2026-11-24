// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-abi-to-llvm --verify-each | FileCheck %s --implicit-check-not="tvm_ffi_abi.upcast" --implicit-check-not="tvm_ffi_abi.downcast"

// Tagged CAny, native function pointers and indexed POD identity conversions.
// Semantic type normalization and folding belong to the dialect fixtures.

// CHECK-LABEL: func.func @any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @any(%arg: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any {
  %wide = tvm_ffi_abi.upcast %arg : !tvm_ffi_abi.any -> !tvm_ffi_abi.any
  %narrow = tvm_ffi_abi.downcast %wide : !tvm_ffi_abi.any -> !tvm_ffi_abi.any
  return %narrow : !tvm_ffi_abi.any
}

// CHECK-LABEL: func.func @function(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @function(%arg: !tvm_ffi_abi.function) -> !tvm_ffi_abi.function {
  %wide = tvm_ffi_abi.upcast %arg : !tvm_ffi_abi.function -> !tvm_ffi_abi.function
  %narrow = tvm_ffi_abi.downcast %wide : !tvm_ffi_abi.function -> !tvm_ffi_abi.function
  return %narrow : !tvm_ffi_abi.function
}

// CHECK-LABEL: func.func @int(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]:
// CHECK-NEXT: return %[[VALUE]]
func.func @int(%arg: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  %wide = tvm_ffi_abi.upcast %arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.int
  %narrow = tvm_ffi_abi.downcast %wide : !tvm_ffi_abi.int -> !tvm_ffi_abi.int
  return %narrow : !tvm_ffi_abi.int
}
