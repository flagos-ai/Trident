// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.any' --implicit-check-not='!tvm_ffi.union<' --implicit-check-not='!tvm_ffi.str' --implicit-check-not='tvm_ffi.assert_type'

// CHECK-LABEL: func.func @narrow_union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
func.func @narrow_union(%value: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %narrow = tvm_ffi.downcast %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %narrow : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// CHECK-LABEL: func.func @unused_string_downcast(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// CHECK-NEXT: return
func.func @unused_string_downcast(%value: !tvm_ffi.any) {
  %string = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.str
  return
}

// CHECK-LABEL: func.func @manual_c_any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
func.func @manual_c_any(%value: !tvm_ffi.c.any) -> !tvm_ffi.c.any {
  return %value : !tvm_ffi.c.any
}

// Result wrappers and unpacked dynamic payloads convert recursively.
// CHECK-LABEL: func.func @result_payload(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.any>) -> !tvm_ffi.c.any
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.c.any>) -> (i1, !tvm_ffi.c.any)
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[PAYLOAD]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[PAYLOAD]] : !tvm_ffi.c.any
func.func @result_payload(%input: !tvm_ffi.monad.result<!tvm_ffi.any>) -> !tvm_ffi.any {
  %ok, %payload = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  return %payload : !tvm_ffi.any
}

// Finite boundary constraints are currently erased to unconstrained CAny.
// CHECK-LABEL: tvm_ffi.func @boundary(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[VALUE]] : !tvm_ffi.c.any -> !tvm_ffi.c.any
// CHECK: tvm_ffi.ObjectIncRef %[[PAYLOAD]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.return %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD]]
tvm_ffi.func @boundary(%value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.str>) -> !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.str> attributes {emit_tvm_ffi_abi} {
  %payload = tvm_ffi.upcast %value : !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.str> -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}
