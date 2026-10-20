// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.downcast'
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// CHECK-LABEL:   func.func @any_and_union(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> (!tvm_ffi.c.array, !tvm_ffi.c.array) {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           return %[[ARG0]], %[[ARG1]] : !tvm_ffi.c.array, !tvm_ffi.c.array
// CHECK-LABEL:   func.func @failed_cleanup() {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @native_function(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.function) -> !tvm_ffi.c.array {
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.function -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectIncRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return %[[VAL_1]] : !tvm_ffi.c.array






// Each tagged Result is retained under its own success condition. Even a
// scalar-only Union uses the Any reference operation's runtime tag check.
// LLVM-LABEL: llvm.func @any_and_union(
func.func @any_and_union(%any: !tvm_ffi.monad.result<!tvm_ffi.any>, %union: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (!tvm_ffi.monad.result<!tvm_ffi.any>, !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) {
  return {tvm_ffi.retain} %any, %union : !tvm_ffi.monad.result<!tvm_ffi.any>, !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>
}

// A failed Result still has a guarded release; cleanup does not fold the branch
// or release its invalid native payload unconditionally.
// LLVM-LABEL: llvm.func @failed_cleanup(
func.func @failed_cleanup() {
  %failed = tvm_ffi.monad.fail {tvm_ffi.cleanup} : !tvm_ffi.monad.result<!tvm_ffi.function>
  return
}

// Function payloads stay unboxed. Export retention precedes local release,
// and generated branches do not introduce another cleanup responsibility.
// LLVM-LABEL: llvm.func @native_function(
func.func @native_function(%input: !tvm_ffi.function) -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  %result = tvm_ffi.monad.pure %input {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.function) -> !tvm_ffi.monad.result<!tvm_ffi.function>
  return {tvm_ffi.retain} %result : !tvm_ffi.monad.result<!tvm_ffi.function>
}
