// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --verify-each | FileCheck %s
// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s

// CHECK-LABEL: func.func @captured(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: tvm_ffi.defer {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @captured(%input: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  return
}

// Even an empty defer is conservatively effectful until explicitly lowered.
// CHECK-LABEL: func.func @empty()
// CHECK: tvm_ffi.defer {
// CHECK-NEXT: tvm_ffi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @empty() {
  tvm_ffi.defer {
    tvm_ffi.defer_yield
  }
  return
}
