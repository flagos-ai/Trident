// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// CHECK-LABEL:   func.func @allocations(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           %[[TENSOR_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tensor.clone %[[ARG0]] : !tvm_ffi.tensor -> !tvm_ffi.tensor
// CHECK:           %[[TENSOR_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tensor.literal dense<1.000000e+00> : tensor<1xf32> : !tvm_ffi.tensor
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "test.identity" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[TENSOR_1]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[TENSOR_0]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return
// CHECK-LABEL:   func.func @containers(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.int 1
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]], %[[CONSTANT_0]]) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[VAL_1]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[VAL_1]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:           %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]](%[[ARG0]]) : (!tvm_ffi.tensor) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_6]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_3]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_8]](%[[VAL_7]], %[[TO_2]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_8]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_9]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @overrides() {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[EXCEPTION_0]] : !tvm_ffi.exception -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:           %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @results(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.tensor
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.int 1
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[UPCAST_1]] : !tvm_ffi.any
// CHECK:           %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:           %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @scalar_and_empty_cleanup() {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.int 7
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant true
// CHECK:           scf.if %[[CONSTANT_2]] {
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return






// Explicit markers cover newly acquired references.
func.func @allocations(%input: !tvm_ffi.tensor) {
  %exception = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %clone = tvm_ffi.tensor.clone %input {tvm_ffi.cleanup} : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %literal = tvm_ffi.tensor.literal dense<1.0> : tensor<1xf32> {tvm_ffi.cleanup} : !tvm_ffi.tensor
  %handle = tvm_ffi.FunctionGetGlobal "test.identity" {tvm_ffi.cleanup} : !tvm_ffi.object
  %called = tvm_ffi.FunctionCall %handle(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
  return {tvm_ffi.retain}
}

// Mixed results acquire credits independently; aliases and scalars stay borrowed.
func.func @containers(%input: !tvm_ffi.tensor) {
  %number = tvm_ffi.constant.int 1
  %tuple = tvm_ffi.tuple.construct(%input, %number) {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>
  %elements:2 = tvm_ffi.tuple.unconstruct %tuple {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>) -> (!tvm_ffi.tensor, !tvm_ffi.int)
  %array = tvm_ffi.array.create(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.array<!tvm_ffi.tensor>
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %array[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.tensor
  %alias = tvm_ffi.upcast %element : !tvm_ffi.tensor -> !tvm_ffi.any
  return {tvm_ffi.retain}
}

// Unmarked references can be transferred; marked references release once.
func.func @overrides() {
  %transferred = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %result = tvm_ffi.monad.pure %transferred {tvm_ffi.cleanup} : (!tvm_ffi.exception) -> !tvm_ffi.monad.result<!tvm_ffi.exception>
  %multiple = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain}
}

// Result cleanup is type-dependent; failed Results never read payloads.
func.func @results(%input: !tvm_ffi.tensor) {
  tvm_ffi.ObjectIncRef %input : !tvm_ffi.tensor
  %number = tvm_ffi.constant.int 1
  %results:2 = tvm_ffi.monad.pure %input, %number {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>)
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return {tvm_ffi.retain}
}

// Scalar and zero-result markers are consumed without reference operations.
func.func @scalar_and_empty_cleanup() {
  %number = tvm_ffi.constant.int 7 {tvm_ffi.cleanup}
  %result = tvm_ffi.monad.pure %number {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.monad.bind : () -> () {
    tvm_ffi.monad.yield {tvm_ffi.retain}
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain}
}
