// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-monad --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.monad.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: func.func @empty() {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// CHECK: scf.if %[[CONSTANT_0]] {
// CHECK: func.call @record() : () -> ()
// CHECK: return
// CHECK-LABEL: func.func @no_inputs(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// CHECK: %[[IF_0:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[CONSTANT_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-DAG: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: scf.yield %[[TUPLE_0]], %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: } else {
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK: %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: scf.yield %[[TUPLE_2]], %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: return %[[IF_0]]#0, %[[IF_0]]#1 : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL: func.func @no_outputs(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_1]]#0 : !tvm_ffi.bool -> i1
// CHECK: %[[ANDI_0:[a-zA-Z0-9_$.-]+]] = arith.andi %[[GET_0]], %[[GET_1]] : i1
// CHECK: scf.if %[[ANDI_0]] {
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_0]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_1]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: func.call @record_values(%[[DOWNCAST_0]], %[[DOWNCAST_1]]) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return

func.func @empty() {
  tvm_ffi.monad.bind : () -> () {
    func.call @record() : () -> ()
    tvm_ffi.monad.yield
  }
  return
}

// No inputs means the continuation executes unconditionally and may produce
// any number of Results.
func.func @no_inputs(%value: !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  %output:2 = tvm_ffi.monad.bind : () -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
    %pure:2 = tvm_ffi.monad.pure %value, %value : (!tvm_ffi.int, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>)
    tvm_ffi.monad.yield %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
}

// Every input must succeed before the side effect executes. The no-output
// failure path needs neither a failed Result nor an else branch.
func.func @no_outputs(%first: !tvm_ffi.monad.result<!tvm_ffi.int>, %second: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  tvm_ffi.monad.bind %first, %second : (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%a: !tvm_ffi.int, %b: !tvm_ffi.int):
    func.call @record_values(%a, %b) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield
  }
  return
}

func.func private @record()
func.func private @record_values(!tvm_ffi.int, !tvm_ffi.int)

// -----

// CHECK-LABEL: func.func @aggregate(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: %[[INITIAL:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[INITIAL_OK:[a-zA-Z0-9_]+]], %[[INITIAL_VALUE:[a-zA-Z0-9_]+]])
// CHECK: scf.yield {test.transfer} %[[INITIAL]]
// CHECK: %[[PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[INPUT]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[PARTS]]#0 : !tvm_ffi.bool -> i1
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PARTS]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]] : !tvm_ffi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]]
// CHECK: %[[BOXED_SUM:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[SUM]] : i64 -> !tvm_ffi.int
// CHECK: %[[WIDE_SUM:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[BOXED_SUM]]
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[YES:[a-zA-Z0-9_]+]], %[[WIDE_SUM]])
// CHECK: scf.yield %[[SUCCESS]]
// CHECK: } else {
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[NO:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool false
// CHECK: %[[WIDE_NONE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: %[[FAILURE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[NO]], %[[WIDE_NONE]])
// CHECK: scf.yield %[[FAILURE]]
// CHECK: return %[[OUTPUT]]
func.func @aggregate(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.int>) {
    %value_boxed = tvm_ffi.to %value : i64 -> !tvm_ffi.int
    %success = tvm_ffi.monad.pure %value_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield {test.transfer} %success : !tvm_ffi.monad.result<!tvm_ffi.int>
  } else {
    %failure = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield %failure : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%payload_boxed: !tvm_ffi.int):
    %payload = tvm_ffi.get %payload_boxed : !tvm_ffi.int -> i64
    %sum = arith.addi %payload, %value : i64
    %sum_boxed = tvm_ffi.to %sum : i64 -> !tvm_ffi.int
    %result = tvm_ffi.monad.pure %sum_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<!tvm_ffi.int>
  }

  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @construct_multiple(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.none,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.function) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-DAG: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG1]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK-DAG: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-DAG: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG2]] : !tvm_ffi.c.function -> !tvm_ffi.any
// CHECK-DAG: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
func.func @construct_multiple(%value: !tvm_ffi.int, %none: !tvm_ffi.none, %function: !tvm_ffi.c.function) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>) {
  %results:3 = tvm_ffi.monad.pure %value, %none, %function : (!tvm_ffi.int, !tvm_ffi.none, !tvm_ffi.c.function) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>)
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

// CHECK-LABEL: func.func @failures() -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
func.func @failures() -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>) {
  %results:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME: %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME: %[[CAPTURE:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[TENSOR_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[TENSOR]]
// CHECK: %[[TENSOR_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR_PARTS]]#0
// CHECK: %[[SCALAR_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[SCALAR]]
// CHECK: %[[SCALAR_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR_PARTS]]#0
// CHECK: %[[BOTH:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_OK]], %[[SCALAR_OK]]
// CHECK: %[[OUTPUTS:[a-zA-Z0-9_]+]]:2 = scf.if %[[BOTH]]
// CHECK: %[[TENSOR_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[TENSOR_PARTS]]#1 : !tvm_ffi.any -> !tvm_ffi.c.tensor
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[SCALAR_PARTS]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]]
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[VALUE]], %[[CAPTURE]]
// CHECK: %[[BOXED_SUM:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[SUM]]
// CHECK: %[[TENSOR_ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TENSOR_VALUE]]
// CHECK: %[[TENSOR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[TENSOR_YES:[a-zA-Z0-9_]+]], %[[TENSOR_ANY]])
// CHECK: %[[SCALAR_ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[BOXED_SUM]]
// CHECK: %[[SCALAR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[SCALAR_YES:[a-zA-Z0-9_]+]], %[[SCALAR_ANY]])
// CHECK: scf.yield %[[TENSOR_RESULT]], %[[SCALAR_RESULT]]
// CHECK: return %[[OUTPUTS]]#0, %[[OUTPUTS]]#1
func.func private @helper(%tensor: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, %scalar: !tvm_ffi.monad.result<!tvm_ffi.int>, %capture: i64) -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  %output:2 = tvm_ffi.monad.bind %tensor, %scalar : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  ^bb0(%payload: !tvm_ffi.c.tensor, %value_boxed: !tvm_ffi.int):
    %value = tvm_ffi.get %value_boxed : !tvm_ffi.int -> i64
    %sum = arith.addi %value, %capture : i64
    %sum_boxed = tvm_ffi.to %sum : i64 -> !tvm_ffi.int
    %pure:2 = tvm_ffi.monad.pure %payload, %sum_boxed : (!tvm_ffi.c.tensor, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>)
    tvm_ffi.monad.yield %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @nested_mixed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME: %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CHECK: %[[INPUT_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[INPUT]]
// CHECK: %[[INPUT_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[INPUT_PARTS]]#0
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if %[[INPUT_OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT_PARTS]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]]
// CHECK: %[[UNIT_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[UNIT]]
// CHECK: %[[UNIT_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[UNIT_PARTS]]#0
// CHECK: %[[INNER:[a-zA-Z0-9_]+]]:2 = scf.if %[[UNIT_OK]]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[UNIT_PARTS]]#1 : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: %[[CAPTURE:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[PAYLOAD]]
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1
// CHECK: return %[[OUTER]]#0, %[[OUTER]]#1
func.func @nested_mixed(%input: !tvm_ffi.monad.result<!tvm_ffi.int>, %unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %outer:2 = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  ^bb0(%payload_boxed: !tvm_ffi.int):
    %payload = tvm_ffi.get %payload_boxed : !tvm_ffi.int -> i64
    %inner:2 = tvm_ffi.monad.bind %unit : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  ^bb_unit(%unit_payload: !tvm_ffi.none):
      %result_none = tvm_ffi.constant.none
      %payload_reboxed = tvm_ffi.to %payload : i64 -> !tvm_ffi.int
      %result:2 = tvm_ffi.monad.pure %payload_reboxed, %result_none : (!tvm_ffi.int, !tvm_ffi.none) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>)
      tvm_ffi.monad.yield %result#0, %result#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
    }
    tvm_ffi.monad.yield %inner#0, %inner#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
  }

  func.return %outer#0, %outer#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: func.func @unpack_function(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK: %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK: return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
func.func @unpack_function(%result: !tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// -----

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[INPUT]]) {arg_attrs = [{test.argument}], res_attrs = [{}, {test.result}]}
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[OK]], %[[VALUE]])
// CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
func.func @invoke(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  %result = tvm_ffi.monad.call @identity(%input) {arg_attrs = [{test.argument}], res_attrs = [{test.result}]} : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
}

// CHECK-LABEL: func.func @invoke_unmarked(
// CHECK: %[[UNMARKED_OK:[a-zA-Z0-9_]+]], %[[UNMARKED_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[UNMARKED_INPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT: %[[UNMARKED_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[UNMARKED_OK]], %[[UNMARKED_VALUE]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-NEXT: return %[[UNMARKED_RESULT]]
func.func @invoke_unmarked(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  %result = tvm_ffi.monad.call @identity(%input) {res_attrs = []} : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
}

// Callee declarations provide the Result payload contracts; packed wrapper
// definitions and declaration reuse are tested by Pipeline/local-call.mlir.
tvm_ffi.func @function_identity(!tvm_ffi.c.function) -> !tvm_ffi.c.function attributes {emit_tvm_ffi_abi}
tvm_ffi.func @identity(!tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi}

// -----

// Monad type conversion supports CFG signatures independently of automatic lifetimes.
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[INPUT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CHECK: ^[[NEXT]](%[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @identity(%[[VALUE]]) : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: return %[[RESULT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
func.func @forward(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  cf.br ^next(%input : !tvm_ffi.monad.result<!tvm_ffi.int>)
^next(%value: !tvm_ffi.monad.result<!tvm_ffi.int>):
  %result = call @identity(%value) : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
}

func.func private @identity(!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
