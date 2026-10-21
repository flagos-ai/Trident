// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield' --implicit-check-not='unrealized_conversion_cast'
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s

// CHECK-LABEL:   func.func @container_continuation(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int) -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.c.any -> !tvm_ffi.c.array
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]] : !tvm_ffi.c.function(%[[DOWNCAST_0]], %[[ARG1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_5]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]] : !tvm_ffi.c.function(%[[CONSTANT_3]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK:             %[[VAL_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_8]] : !tvm_ffi.c.function(%[[CONSTANT_5]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_9]] : !tvm_ffi.c.array
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @nested_containers(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: i64,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.c.any -> !tvm_ffi.c.array
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:             %[[TO_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_3]] : i64 -> !tvm_ffi.int
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]] : !tvm_ffi.c.function(%[[DOWNCAST_0]], %[[TO_2]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:             %[[TO_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_4]] : i64 -> !tvm_ffi.int
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]] : !tvm_ffi.c.function(%[[DOWNCAST_0]], %[[TO_3]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK:             %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG3]] -> (!tvm_ffi.c.array) {
// CHECK:               scf.yield %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:             } else {
// CHECK:               %[[VAL_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:               %[[VAL_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_8]] : !tvm_ffi.c.function(%[[VAL_7]]) : (!tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK:               scf.yield %[[VAL_9]] : !tvm_ffi.c.array
// CHECK:             %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:             %[[TO_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_6]] : i64 -> !tvm_ffi.int
// CHECK:             %[[VAL_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:             %[[VAL_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_10]] : !tvm_ffi.c.function(%[[ARG1]], %[[TO_4]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:             %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:             %[[TO_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_7]] : i64 -> !tvm_ffi.int
// CHECK:             %[[VAL_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:             %[[VAL_13:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_12]] : !tvm_ffi.c.function(%[[ARG1]], %[[TO_5]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK:             %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_11]] : !tvm_ffi.bool -> i1
// CHECK:             %[[IF_2:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_1]] -> (!tvm_ffi.c.array) {
// CHECK:               %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_13]] : !tvm_ffi.c.any -> !tvm_ffi.int
// CHECK:               %[[TO_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[ARG2]] : i64 -> !tvm_ffi.int
// CHECK:               %[[VAL_14:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:               %[[VAL_15:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_14]] : !tvm_ffi.c.function(%[[IF_1]], %[[TO_6]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK:               %[[VAL_16:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:               %[[VAL_17:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_16]] : !tvm_ffi.c.function(%[[VAL_15]], %[[DOWNCAST_1]], %[[VAL_7]]) : (!tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK-DAG:               %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_17]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG:               %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:               %[[VAL_18:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:               %[[VAL_19:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_18]] : !tvm_ffi.c.function(%[[CONSTANT_8]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:               scf.yield %[[VAL_19]] : !tvm_ffi.c.array
// CHECK:             } else {
// CHECK:               %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:               %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:               %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_9]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK:               %[[VAL_20:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:               %[[VAL_21:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_20]] : !tvm_ffi.c.function(%[[CONSTANT_10]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:               scf.yield %[[VAL_21]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[IF_2]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_11]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK:             %[[VAL_22:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_23:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_22]] : !tvm_ffi.c.function(%[[CONSTANT_12]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_23]] : !tvm_ffi.c.array
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array

// Container calls must stay inside the successful continuation, including calls
// whose operands are lexical captures rather than bind payloads.
func.func @container_continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>, %capture: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>> {
  ^bb0(%array: !tvm_ffi.array<!tvm_ffi.int>):
    %tuple = tvm_ffi.tuple.construct(%array, %capture) : (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>
    %pure = tvm_ffi.monad.pure %tuple : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
  }
  return %result : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
}

// Nested continuations combine remapped block arguments, lexical captures and
// structural SCF conversion. No semantic container cast may escape conversion.
func.func @nested_containers(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>, %scalar: !tvm_ffi.monad.result<!tvm_ffi.int>, %index: i64, %predicate: i1) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  %outer = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  ^bb0(%tuple: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>):
    %array, %capture = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int)
    %chosen = scf.if %predicate -> (!tvm_ffi.array<!tvm_ffi.int>) {
      scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
    } else {
      %fallback = tvm_ffi.array.create(%capture) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
      scf.yield %fallback : !tvm_ffi.array<!tvm_ffi.int>
    }
    %inner = tvm_ffi.monad.bind %scalar : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
    ^bb1(%value: !tvm_ffi.int):
      %item = tvm_ffi.array.get %chosen[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
      %result = tvm_ffi.array.create(%item, %value, %capture) : (!tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
      %pure = tvm_ffi.monad.pure %result : (!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
      tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
    }
    tvm_ffi.monad.yield %inner : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
  }
  return %outer : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
}
