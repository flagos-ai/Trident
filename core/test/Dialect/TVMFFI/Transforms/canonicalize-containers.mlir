// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s

// CHECK-LABEL: func.func @direct_roundtrip(
// CHECK-SAME: %[[INTEGER:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[BOOLEAN:[a-zA-Z0-9_]+]]: !tvm_ffi.bool)
// CHECK-NOT: tvm_ffi.tuple.construct
// CHECK-NOT: tvm_ffi.tuple.unconstruct
// CHECK: return %[[INTEGER]], %[[BOOLEAN]]
func.func @direct_roundtrip(%integer: !tvm_ffi.int,
    %boolean: !tvm_ffi.bool) -> (!tvm_ffi.int, !tvm_ffi.bool) {
  %tuple = tvm_ffi.tuple.construct(%integer, %boolean) :
      (!tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>
  %first, %second = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>) -> (!tvm_ffi.int, !tvm_ffi.bool)
  return %first, %second : !tvm_ffi.int, !tvm_ffi.bool
}

// Empty projections and their dead tuple need no runtime container operations.
// CHECK-LABEL: func.func @empty_tuple()
// CHECK-NEXT: return
func.func @empty_tuple() {
  %tuple = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<>) -> ()
  return
}

// Multiple unpack users do not prevent forwarding or dead construct cleanup.
// CHECK-LABEL: func.func @multiple_unconstructs(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: return %[[VALUE]], %[[VALUE]]
func.func @multiple_unconstructs(%value: !tvm_ffi.int) -> (!tvm_ffi.int, !tvm_ffi.int) {
  %tuple = tvm_ffi.tuple.construct(%value) :
      (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %first = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  %second = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  return %first, %second : !tvm_ffi.int, !tvm_ffi.int
}

// CHECK-LABEL: func.func @object_roundtrip(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.any)
// CHECK-NOT: tvm_ffi.tuple.construct
// CHECK-NOT: tvm_ffi.tuple.unconstruct
// CHECK: return %[[VALUE]]
func.func @object_roundtrip(%value: !tvm_ffi.any) -> !tvm_ffi.any {
  %tuple = tvm_ffi.tuple.construct(%value) :
      (!tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.any>
  %result = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.any>) -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// Forward unpacked elements while retaining the observable tuple.
// CHECK-LABEL: func.func @shared_tuple(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[VALUE]])
// CHECK-NEXT: return %[[TUPLE]], %[[VALUE]]
func.func @shared_tuple(%value: !tvm_ffi.int) ->
    (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.int) {
  %tuple = tvm_ffi.tuple.construct(%value) :
      (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %result = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  return %tuple, %result : !tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.int
}
