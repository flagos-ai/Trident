// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --ownership-deallocation --verify-each | FileCheck %s

// CHECK-LABEL: func.func @array(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.array
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.array
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK: return %[[NARROW]]
func.func @array(%input: !tvm_ffi.array) -> !tvm_ffi.array {
  %wide = tvm_ffi.upcast %input : !tvm_ffi.array -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.array
  return %narrow : !tvm_ffi.array
}

// CHECK-LABEL: func.func @exception(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.exception
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.exception
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK: return %[[NARROW]]
func.func @exception(%input: !tvm_ffi.exception) -> !tvm_ffi.exception {
  %wide = tvm_ffi.upcast %input : !tvm_ffi.exception -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.exception
  return %narrow : !tvm_ffi.exception
}

// CHECK-LABEL: func.func @tensor(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.tensor
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK: return %[[NARROW]]
func.func @tensor(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %wide = tvm_ffi.upcast %input : !tvm_ffi.tensor -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.tensor
  return %narrow : !tvm_ffi.tensor
}

