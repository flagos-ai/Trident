// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s

// CHECK-LABEL: func.func @direct_roundtrip(
// CHECK-SAME: %[[INTEGER:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[BOOLEAN:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool)
// CHECK-NOT: tvm_ffi_api.tuple.construct
// CHECK-NOT: tvm_ffi_api.tuple.unconstruct
// CHECK: return %[[INTEGER]], %[[BOOLEAN]]
func.func @direct_roundtrip(%integer: !tvm_ffi_abi.int,
    %boolean: !tvm_ffi_abi.bool) -> (!tvm_ffi_abi.int, !tvm_ffi_abi.bool) {
  %tuple = tvm_ffi_api.tuple.construct(%integer, %boolean) :
      (!tvm_ffi_abi.int, !tvm_ffi_abi.bool) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.bool>
  %first, %second = tvm_ffi_api.tuple.unconstruct %tuple :
      (!tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.bool>) -> (!tvm_ffi_abi.int, !tvm_ffi_abi.bool)
  return %first, %second : !tvm_ffi_abi.int, !tvm_ffi_abi.bool
}

// Empty projections and their dead tuple need no runtime container operations.
// CHECK-LABEL: func.func @empty_tuple()
// CHECK-NEXT: return
func.func @empty_tuple() {
  %tuple = tvm_ffi_api.tuple.construct() : () -> !tvm_ffi_api.tuple<>
  tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<>) -> ()
  return
}

// Multiple unpack users do not prevent forwarding or dead construct cleanup.
// CHECK-LABEL: func.func @multiple_unconstructs(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: return %[[VALUE]], %[[VALUE]]
func.func @multiple_unconstructs(%value: !tvm_ffi_abi.int) -> (!tvm_ffi_abi.int, !tvm_ffi_abi.int) {
  %tuple = tvm_ffi_api.tuple.construct(%value) :
      (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
  %first = tvm_ffi_api.tuple.unconstruct %tuple :
      (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int
  %second = tvm_ffi_api.tuple.unconstruct %tuple :
      (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int
  return %first, %second : !tvm_ffi_abi.int, !tvm_ffi_abi.int
}

// CHECK-LABEL: func.func @object_roundtrip(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// CHECK-NOT: tvm_ffi_api.tuple.construct
// CHECK-NOT: tvm_ffi_api.tuple.unconstruct
// CHECK: return %[[VALUE]]
func.func @object_roundtrip(%value: !tvm_ffi_api.any) -> !tvm_ffi_api.any {
  %tuple = tvm_ffi_api.tuple.construct(%value) :
      (!tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_api.any>
  %result = tvm_ffi_api.tuple.unconstruct %tuple :
      (!tvm_ffi_api.tuple<!tvm_ffi_api.any>) -> !tvm_ffi_api.any
  return %result : !tvm_ffi_api.any
}

// Forward unpacked elements while retaining the observable tuple.
// CHECK-LABEL: func.func @shared_tuple(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[VALUE]])
// CHECK-NEXT: return %[[TUPLE]], %[[VALUE]]
func.func @shared_tuple(%value: !tvm_ffi_abi.int) ->
    (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.int) {
  %tuple = tvm_ffi_api.tuple.construct(%value) :
      (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
  %result = tvm_ffi_api.tuple.unconstruct %tuple :
      (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int
  return %tuple, %result : !tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.int
}
