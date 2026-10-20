// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not='llvm.alloca' --implicit-check-not='!tvm_ffi.tuple'
// RUN: trident-core-opt %s --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s --check-prefix=CLEANUP --implicit-check-not='tvm_ffi.cleanup' --implicit-check-not='tvm_ffi.retain'

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[INPUT]], %[[SECOND]]) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}], tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK-NEXT: return {tvm_ffi.retain} %[[OK]], %[[VALUE]]
// CLEANUP-LABEL: func.func @invoke(
// CLEANUP: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.any
// CLEANUP-NEXT: return %[[OK]], %[[VALUE]]
// CHECK-LABEL: func.func @invoke_empty(
// CHECK-NEXT: %[[EMPTY_OK:[a-zA-Z0-9_]+]], %[[EMPTY_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @empty() {tvm_ffi.cleanup} : () -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK-NEXT: return {tvm_ffi.retain} %[[EMPTY_OK]], %[[EMPTY_VALUE]]
// CHECK-LABEL: tvm_ffi.func @identity(
// CHECK-SAME: !tvm_ffi.tensor,
// CHECK-SAME: !tvm_ffi.int
// CHECK-LABEL: tvm_ffi.func @empty(

func.func @invoke(%input: !tvm_ffi.tensor, %second: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @identity(%input, %second) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}], tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return {tvm_ffi.retain} %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
func.func @invoke_empty() -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @empty() {tvm_ffi.cleanup} : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return {tvm_ffi.retain} %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
tvm_ffi.func @identity(%input: !tvm_ffi.tensor, %second: !tvm_ffi.int) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %any = tvm_ffi.upcast %input : !tvm_ffi.tensor -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return {tvm_ffi.retain} %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}
tvm_ffi.func @empty() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi} {
  %none = tvm_ffi.constant.none
  %any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}

// CHECK-LABEL: func.func @invoke_container(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: %[[CONTAINER_OK:[a-zA-Z0-9_]+]], %[[CONTAINER_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @container(%[[ARG]]) {test.attribute} : (!tvm_ffi.c.array) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK-NEXT: return %[[CONTAINER_OK]], %[[CONTAINER_VALUE]]
func.func @invoke_container(%arg: !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @container(%arg) {test.attribute} : (!tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
// CHECK-LABEL: tvm_ffi.func @container(!tvm_ffi.c.array) -> !tvm_ffi.c.array
 tvm_ffi.func @container(!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int> attributes {emit_tvm_ffi_abi}
