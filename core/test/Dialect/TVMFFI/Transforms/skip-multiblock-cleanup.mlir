// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t.once
// RUN: trident-core-opt %t.once --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t.once -

// CHECK-LABEL:   func.func private @external() -> !tvm_ffi.tensor
// CHECK-LABEL:   func.func @multiple_blocks(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi.exception {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.exception) {
// CHECK:             %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:           } else {
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           cf.br ^[[BLOCK_bb1:bb[0-9]+]]
// CHECK:           return {tvm_ffi.retain} %[[IF_0]] : !tvm_ffi.exception
// CHECK-LABEL:   func.func @pure_in_multiple_blocks(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) -> !tvm_ffi.c.array {
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_0]], %[[UPCAST_0]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           cf.br ^[[BLOCK_bb1:bb[0-9]+]](%[[VAL_1]] : !tvm_ffi.c.array)
// CHECK:         ^[[BLOCK_bb1]](%[[VAL_2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array):
// CHECK:           return {tvm_ffi.retain} %[[VAL_2]] : !tvm_ffi.c.array






// An empty function region requires no processing.
func.func private @external() -> !tvm_ffi.tensor

// Direct operations in the multi-block function remain unchanged. Each visited
// nested single-block region independently acquires exports and releases locals.
func.func @multiple_blocks(%predicate: i1) -> !tvm_ffi.exception {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %output = scf.if %predicate -> !tvm_ffi.exception {
    %inner = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %inner : !tvm_ffi.exception
  } else {
    scf.yield {tvm_ffi.retain} %local : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  cf.br ^exit
^exit:
  return {tvm_ffi.retain} %output : !tvm_ffi.exception
}

// Pure retains do not require a single-block region. Local cleanup and export
// markers remain pending, and a second pass must not retain the payload again.
func.func @pure_in_multiple_blocks(%input: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %result = tvm_ffi.monad.pure %input {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  cf.br ^exit(%result : !tvm_ffi.monad.result<!tvm_ffi.tensor>)
^exit(%output: !tvm_ffi.monad.result<!tvm_ffi.tensor>):
  return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
