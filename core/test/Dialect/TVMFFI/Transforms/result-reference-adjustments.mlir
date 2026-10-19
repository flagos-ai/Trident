// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.downcast'
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --materialize-tvm-ffi-cleanup --verify-each | diff %t -
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// Each tagged Result is retained under its own success condition. Even a
// scalar-only Union uses the Any reference operation's runtime tag check.
// CHECK-LABEL: func.func @any_and_union(
// CHECK-SAME: %[[ANY:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.any>, %[[UNION:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>
// CHECK-NEXT: %[[ANY_OK:[a-zA-Z0-9_]+]], %[[ANY_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[ANY]]
// CHECK-NEXT: scf.if %[[ANY_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ANY_PAYLOAD]] : !tvm_ffi.any
// CHECK-NEXT: }
// CHECK-NEXT: %[[UNION_OK:[a-zA-Z0-9_]+]], %[[UNION_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[UNION]]
// CHECK-NEXT: scf.if %[[UNION_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[UNION_PAYLOAD]] : !tvm_ffi.any
// CHECK-NEXT: }
// CHECK-NEXT: return %[[ANY]], %[[UNION]]
// LLVM-LABEL: llvm.func @any_and_union(
func.func @any_and_union(%any: !tvm_ffi.monad.result<!tvm_ffi.any>, %union: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (!tvm_ffi.monad.result<!tvm_ffi.any>, !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) {
  return {tvm_ffi.retain} %any, %union : !tvm_ffi.monad.result<!tvm_ffi.any>, !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>
}

// A failed Result still has a guarded release; cleanup does not fold the branch
// or release its invalid native payload unconditionally.
// CHECK-LABEL: func.func @failed_cleanup(
// CHECK-NEXT: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[FAILED]]
// CHECK-NEXT: scf.if %[[OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD]] : !tvm_ffi.object
// CHECK-NEXT: }
// CHECK-NEXT: return
// LLVM-LABEL: llvm.func @failed_cleanup(
func.func @failed_cleanup() {
  %failed = tvm_ffi.monad.fail {tvm_ffi.cleanup} : !tvm_ffi.monad.result<!tvm_ffi.object>
  return
}

// Native Object payloads stay unboxed. Export retention precedes local release,
// and generated branches do not introduce another cleanup responsibility.
// CHECK-LABEL: func.func @native_object(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.object
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.object
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[INPUT]]
// CHECK-NEXT: %[[RETAIN_OK:[a-zA-Z0-9_]+]], %[[RETAIN_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[RETAIN_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RETAIN_PAYLOAD]] : !tvm_ffi.object
// CHECK-NEXT: }
// CHECK-NEXT: %[[RELEASE_OK:[a-zA-Z0-9_]+]], %[[RELEASE_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[RELEASE_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RELEASE_PAYLOAD]] : !tvm_ffi.object
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]]
// LLVM-LABEL: llvm.func @native_object(
func.func @native_object(%input: !tvm_ffi.object) -> !tvm_ffi.monad.result<!tvm_ffi.object> {
  %result = tvm_ffi.monad.pure %input {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.object) -> !tvm_ffi.monad.result<!tvm_ffi.object>
  return {tvm_ffi.retain} %result : !tvm_ffi.monad.result<!tvm_ffi.object>
}
