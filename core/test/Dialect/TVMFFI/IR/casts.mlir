// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | FileCheck %s

// CHECK-LABEL: func.func @any_to_int(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: return %[[RESULT]]
func.func @any_to_int(%arg: !tvm_ffi.any) -> !tvm_ffi.int {
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @any_to_union(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
// CHECK: return %[[RESULT]]
func.func @any_to_union(%arg: !tvm_ffi.any) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
}

// CHECK-LABEL: func.func @union_to_int(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.int
// CHECK: return %[[RESULT]]
func.func @union_to_int(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.int {
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @union_to_subset(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
// CHECK: return %[[RESULT]]
func.func @union_to_subset(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
}

// CHECK-LABEL: func.func @union_to_any(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.any
// CHECK: return %[[RESULT]]
func.func @union_to_any(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.any {
  %result = tvm_ffi.upcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// CHECK-LABEL: func.func @union_to_superset(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
// CHECK: return %[[RESULT]]
func.func @union_to_superset(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> {
  %result = tvm_ffi.upcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
}

// CHECK-LABEL: func.func @any_identity(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT:[a-zA-Z0-9_]+]] : !tvm_ffi.any -> !tvm_ffi.any
// CHECK: return %[[RESULT]]
func.func @any_identity(%arg: !tvm_ffi.any) -> !tvm_ffi.any {
  %result = tvm_ffi.upcast %arg : !tvm_ffi.any -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

