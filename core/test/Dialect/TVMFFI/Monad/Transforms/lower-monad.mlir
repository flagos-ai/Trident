// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-monad --verify-each | FileCheck %s --implicit-check-not='tvm_ffi_monad.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: func.func @empty() {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// CHECK: scf.if %[[CONSTANT_0]] {
// CHECK: func.call @record() : () -> ()
// CHECK: return
// CHECK-LABEL: func.func @no_inputs(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.int) -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// CHECK: %[[IF_0:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[CONSTANT_0]] -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[ARG0]] : !tvm_ffi_abi.int -> !tvm_ffi_api.any
// CHECK-DAG: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-DAG: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[ARG0]] : !tvm_ffi_abi.int -> !tvm_ffi_api.any
// CHECK-DAG: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: scf.yield %[[TUPLE_0]], %[[TUPLE_1]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: } else {
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[CONSTANT_3]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[CONSTANT_3]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK: %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: scf.yield %[[TUPLE_2]], %[[TUPLE_3]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: return %[[IF_0]]#0, %[[IF_0]]#1 : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-LABEL: func.func @no_outputs(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) {
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[ARG0]] : (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.get %[[TUPLE_0]]#0 : !tvm_ffi_abi.bool -> i1
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[ARG1]] : (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
// CHECK: %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.get %[[TUPLE_1]]#0 : !tvm_ffi_abi.bool -> i1
// CHECK: %[[ANDI_0:[a-zA-Z0-9_$.-]+]] = arith.andi %[[GET_0]], %[[GET_1]] : i1
// CHECK: scf.if %[[ANDI_0]] {
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.downcast %[[TUPLE_0]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// CHECK: %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.downcast %[[TUPLE_1]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// CHECK: func.call @record_values(%[[DOWNCAST_0]], %[[DOWNCAST_1]]) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return

func.func @empty() {
  tvm_ffi_monad.bind : () -> () {
    func.call @record() : () -> ()
    tvm_ffi_monad.yield
  }
  return
}

// No inputs means the continuation executes unconditionally and may produce
// any number of Results.
func.func @no_inputs(%value: !tvm_ffi_abi.int) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
  %output:2 = tvm_ffi_monad.bind : () -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
    %pure_0 = tvm_ffi_monad.pure %value : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    %pure_1 = tvm_ffi_monad.pure %value : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    tvm_ffi_monad.yield %pure_0, %pure_1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }
  return %output#0, %output#1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// Every input must succeed before the side effect executes. The no-output
// failure path needs neither a failed Result nor an else branch.
func.func @no_outputs(%first: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %second: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
  tvm_ffi_monad.bind %first, %second : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> () {
  ^bb0(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int):
    func.call @record_values(%a, %b) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    tvm_ffi_monad.yield
  }
  return
}

func.func private @record()
func.func private @record_values(!tvm_ffi_abi.int, !tvm_ffi_abi.int)

// -----

// CHECK-LABEL: func.func @aggregate(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: %[[INITIAL:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[INITIAL_OK:[a-zA-Z0-9_]+]], %[[INITIAL_VALUE:[a-zA-Z0-9_]+]])
// CHECK: scf.yield {test.transfer} %[[INITIAL]]
// CHECK: %[[PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[INPUT]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[PARTS]]#0 : !tvm_ffi_abi.bool -> i1
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[PARTS]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[BOXED]] : !tvm_ffi_abi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]]
// CHECK: %[[BOXED_SUM:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[SUM]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[WIDE_SUM:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[BOXED_SUM]]
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[YES:[a-zA-Z0-9_]+]], %[[WIDE_SUM]])
// CHECK: scf.yield %[[SUCCESS]]
// CHECK: } else {
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[NO:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[WIDE_NONE:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[NONE]]
// CHECK: %[[FAILURE:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[NO]], %[[WIDE_NONE]])
// CHECK: scf.yield %[[FAILURE]]
// CHECK: return %[[OUTPUT]]
func.func @aggregate(%value: i64, %ready: i1) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  %input = scf.if %ready -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
    %value_boxed = tvm_ffi_abi.to %value : i64 -> !tvm_ffi_abi.int
    %success = tvm_ffi_monad.pure %value_boxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    scf.yield {test.transfer} %success : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  } else {
    %failure = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    scf.yield %failure : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }
  %output = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%payload_boxed: !tvm_ffi_abi.int):
    %payload = tvm_ffi_abi.get %payload_boxed : !tvm_ffi_abi.int -> i64
    %sum = arith.addi %payload, %value : i64
    %sum_boxed = tvm_ffi_abi.to %sum : i64 -> !tvm_ffi_abi.int
    %result = tvm_ffi_monad.pure %sum_boxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    tvm_ffi_monad.yield %result : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }

  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// CHECK-LABEL: func.func @construct_multiple(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.int,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.none,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.function) -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[ARG0]] : !tvm_ffi_abi.int -> !tvm_ffi_api.any
// CHECK-DAG: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-DAG: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[ARG1]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK-DAG: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_1]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-DAG: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[ARG2]] : !tvm_ffi_abi.function -> !tvm_ffi_api.any
// CHECK-DAG: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_2]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
func.func @construct_multiple(%value: !tvm_ffi_abi.int, %none: !tvm_ffi_abi.none, %function: !tvm_ffi_abi.function) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>, !tvm_ffi_monad.result<!tvm_ffi_abi.function>) {
  %results_0 = tvm_ffi_monad.pure %value : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  %results_1 = tvm_ffi_monad.pure %none : (!tvm_ffi_abi.none) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  %results_2 = tvm_ffi_monad.pure %function : (!tvm_ffi_abi.function) -> !tvm_ffi_monad.result<!tvm_ffi_abi.function>
  return %results_0, %results_1, %results_2 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>, !tvm_ffi_monad.result<!tvm_ffi_abi.function>
}

// CHECK-LABEL: func.func @failures() -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[CONSTANT_0]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[CONSTANT_0]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.upcast %[[CONSTANT_0]] : !tvm_ffi_abi.none -> !tvm_ffi_api.any
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_api.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_2]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>, !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
func.func @failures() -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.function>) {
  %results:3 = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.function>
  return %results#0, %results#1, %results#2 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.function>
}

// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>,
// CHECK-SAME: %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>,
// CHECK-SAME: %[[CAPTURE:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[TENSOR_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[TENSOR]]
// CHECK: %[[TENSOR_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR_PARTS]]#0
// CHECK: %[[SCALAR_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[SCALAR]]
// CHECK: %[[SCALAR_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[SCALAR_PARTS]]#0
// CHECK: %[[BOTH:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_OK]], %[[SCALAR_OK]]
// CHECK: %[[OUTPUTS:[a-zA-Z0-9_]+]]:2 = scf.if %[[BOTH]]
// CHECK: %[[TENSOR_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[TENSOR_PARTS]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.tensor
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[SCALAR_PARTS]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[BOXED]]
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[VALUE]], %[[CAPTURE]]
// CHECK: %[[BOXED_SUM:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[SUM]]
// CHECK: %[[TENSOR_ANY:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[TENSOR_VALUE]]
// CHECK: %[[TENSOR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[TENSOR_YES:[a-zA-Z0-9_]+]], %[[TENSOR_ANY]])
// CHECK: %[[SCALAR_ANY:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[BOXED_SUM]]
// CHECK: %[[SCALAR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[SCALAR_YES:[a-zA-Z0-9_]+]], %[[SCALAR_ANY]])
// CHECK: scf.yield %[[TENSOR_RESULT]], %[[SCALAR_RESULT]]
// CHECK: return %[[OUTPUTS]]#0, %[[OUTPUTS]]#1
func.func private @helper(%tensor: !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, %scalar: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %capture: i64) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
  %output:2 = tvm_ffi_monad.bind %tensor, %scalar : (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
  ^bb0(%payload: !tvm_ffi_abi.tensor, %value_boxed: !tvm_ffi_abi.int):
    %value = tvm_ffi_abi.get %value_boxed : !tvm_ffi_abi.int -> i64
    %sum = arith.addi %value, %capture : i64
    %sum_boxed = tvm_ffi_abi.to %sum : i64 -> !tvm_ffi_abi.int
    %pure_0 = tvm_ffi_monad.pure %payload : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
    %pure_1 = tvm_ffi_monad.pure %sum_boxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    tvm_ffi_monad.yield %pure_0, %pure_1 : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }
  return %output#0, %output#1 : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// CHECK-LABEL: func.func @nested_mixed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>,
// CHECK-SAME: %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>)
// CHECK: %[[INPUT_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[INPUT]]
// CHECK: %[[INPUT_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[INPUT_PARTS]]#0
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if %[[INPUT_OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT_PARTS]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[BOXED]]
// CHECK: %[[UNIT_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[UNIT]]
// CHECK: %[[UNIT_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[UNIT_PARTS]]#0
// CHECK: %[[INNER:[a-zA-Z0-9_]+]]:2 = scf.if %[[UNIT_OK]]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[UNIT_PARTS]]#1 : !tvm_ffi_api.any -> !tvm_ffi_abi.none
// CHECK: %[[CAPTURE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[PAYLOAD]]
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1
// CHECK: return %[[OUTER]]#0, %[[OUTER]]#1
func.func @nested_mixed(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %unit: !tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
  %outer:2 = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
  ^bb0(%payload_boxed: !tvm_ffi_abi.int):
    %payload = tvm_ffi_abi.get %payload_boxed : !tvm_ffi_abi.int -> i64
    %inner:2 = tvm_ffi_monad.bind %unit : (!tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
  ^bb_unit(%unit_payload: !tvm_ffi_abi.none):
      %result_none = tvm_ffi_abi.constant.none
      %payload_reboxed = tvm_ffi_abi.to %payload : i64 -> !tvm_ffi_abi.int
      %result_0 = tvm_ffi_monad.pure %payload_reboxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
      %result_1 = tvm_ffi_monad.pure %result_none : (!tvm_ffi_abi.none) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
      tvm_ffi_monad.yield %result_0, %result_1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    }
    tvm_ffi_monad.yield %inner#0, %inner#1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  }

  func.return %outer#0, %outer#1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
}

// CHECK-LABEL: func.func @unpack_function(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) -> (i1, !tvm_ffi_api.any) {
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[ARG0]] : (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.get %[[TUPLE_0]]#0 : !tvm_ffi_abi.bool -> i1
// CHECK: return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi_api.any
func.func @unpack_function(%result: !tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> (i1, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_monad.unpack %result : (!tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> (i1, !tvm_ffi_api.any)
  return %ok, %value : i1, !tvm_ffi_api.any
}

// -----

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_api.call @identity(%[[INPUT]]) {arg_attrs = [{test.argument}], res_attrs = [{}, {test.result}]}
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[OK]], %[[VALUE]])
// CHECK-NEXT: return %[[RESULT]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
func.func @invoke(%input: !tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor> {
  %result = tvm_ffi_monad.call @identity(%input) {arg_attrs = [{test.argument}], res_attrs = [{test.result}]} : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
}

// CHECK-LABEL: func.func @invoke_unmarked(
// CHECK: %[[UNMARKED_OK:[a-zA-Z0-9_]+]], %[[UNMARKED_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_api.call @identity(%[[UNMARKED_INPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT: %[[UNMARKED_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[UNMARKED_OK]], %[[UNMARKED_VALUE]]) : (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK-NEXT: return %[[UNMARKED_RESULT]]
func.func @invoke_unmarked(%input: !tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor> {
  %result = tvm_ffi_monad.call @identity(%input) {res_attrs = []} : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
}

// Callee declarations provide the Result payload contracts; packed wrapper
// definitions and declaration reuse are tested by Pipeline/local-call.mlir.
tvm_ffi_api.func @function_identity(!tvm_ffi_abi.function) -> !tvm_ffi_abi.function
tvm_ffi_api.func @identity(!tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor

// -----

// Monad type conversion supports CFG signatures independently of automatic lifetimes.
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>)
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[INPUT]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>)
// CHECK: ^[[NEXT]](%[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>):
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @identity(%[[VALUE]]) : (!tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
// CHECK: return %[[RESULT]] : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
func.func @forward(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  cf.br ^next(%input : !tvm_ffi_monad.result<!tvm_ffi_abi.int>)
^next(%value: !tvm_ffi_monad.result<!tvm_ffi_abi.int>):
  %result = call @identity(%value) : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

func.func private @identity(!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
