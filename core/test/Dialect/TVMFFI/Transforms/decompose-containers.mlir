// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.monad.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: i64,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi.c.array {
// CHECK: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK: %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.c.any -> !tvm_ffi.c.array
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_3]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]] : !tvm_ffi.c.function(%[[DOWNCAST_0]], %[[TO_2]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_4]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]] : !tvm_ffi.c.function(%[[DOWNCAST_0]], %[[TO_3]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG3]] -> (!tvm_ffi.c.array) {
// CHECK: scf.yield %[[VAL_5]] : !tvm_ffi.c.array
// CHECK: } else {
// CHECK: %[[VAL_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_8]] : !tvm_ffi.c.function(%[[VAL_7]]) : (!tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: scf.yield %[[VAL_9]] : !tvm_ffi.c.array
// CHECK: %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_6]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_10]] : !tvm_ffi.c.function(%[[ARG1]], %[[TO_4]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK: %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_7]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_13:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_12]] : !tvm_ffi.c.function(%[[ARG1]], %[[TO_5]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK: %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_11]] : !tvm_ffi.bool -> i1
// CHECK: %[[IF_2:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_1]] -> (!tvm_ffi.c.array) {
// CHECK: %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_13]] : !tvm_ffi.c.any -> !tvm_ffi.int
// CHECK: %[[TO_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[ARG2]] : i64 -> !tvm_ffi.int
// CHECK: %[[VAL_14:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK: %[[VAL_15:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_14]] : !tvm_ffi.c.function(%[[IF_1]], %[[TO_6]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: %[[VAL_16:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_17:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_16]] : !tvm_ffi.c.function(%[[VAL_15]], %[[DOWNCAST_1]], %[[VAL_7]]) : (!tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_17]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG: %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[VAL_18:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_19:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_18]] : !tvm_ffi.c.function(%[[CONSTANT_8]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK: scf.yield %[[VAL_19]] : !tvm_ffi.c.array
// CHECK: } else {
// CHECK: %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK: %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_9]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK: %[[VAL_20:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_21:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_20]] : !tvm_ffi.c.function(%[[CONSTANT_10]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK: scf.yield %[[VAL_21]] : !tvm_ffi.c.array
// CHECK: scf.yield %[[IF_2]] : !tvm_ffi.c.array
// CHECK: } else {
// CHECK: %[[CONSTANT_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK: %[[CONSTANT_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_11]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK: %[[VAL_22:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_23:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_22]] : !tvm_ffi.c.function(%[[CONSTANT_12]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK: scf.yield %[[VAL_23]] : !tvm_ffi.c.array
// CHECK: return %[[IF_0]] : !tvm_ffi.c.array

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

// -----

// CHECK-LABEL: func.func @casts(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.c.array -> !tvm_ffi.c.any {test.keep = "wide"}
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[UPCAST_0]] : !tvm_ffi.c.any -> !tvm_ffi.c.array {test.types = [!tvm_ffi.c.array]}
// CHECK: return %[[DOWNCAST_0]] : !tvm_ffi.c.array
// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK: %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK: tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.c.function
// CHECK: return %[[VAL_1]] : !tvm_ffi.c.array
// CHECK-LABEL: func.func @opaque_passthrough(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.any,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.exception,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.object,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.any) -> (!tvm_ffi.c.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.c.any) {
// CHECK: return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !tvm_ffi.c.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.c.any

func.func @casts(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.array<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.any {test.keep = "wide"}
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.array<!tvm_ffi.int> {test.types = [!tvm_ffi.tuple<!tvm_ffi.int>]}
  return %narrow : !tvm_ffi.array<!tvm_ffi.int>
}

func.func @nested_containers(%nested: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>> {
  %result = tvm_ffi.monad.pure %nested {test.types = [!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int>>]} : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>>
}

func.func @opaque_passthrough(%any: !tvm_ffi.any, %exception: !tvm_ffi.c.exception, %object: !tvm_ffi.object, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> (!tvm_ffi.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  return %any, %exception, %object, %union : !tvm_ffi.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// -----

// CHECK-LABEL: func.func @container_attributes(
// CHECK-SAME: attributes {test.types = [!tvm_ffi.c.array, {nested = !tvm_ffi.c.array}]}
func.func @container_attributes() attributes {test.types = [!tvm_ffi.array<!tvm_ffi.int>, {nested = !tvm_ffi.tuple<!tvm_ffi.int>}]} {
  return
}

// CHECK-LABEL: func.func @create_array(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[CALLED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[VALUE]]) : (!tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]] : !tvm_ffi.c.function
// CHECK: return %[[CALLED]]
func.func @create_array(%value: !tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int> {
  %result = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
  return %result : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[EMPTY_HANDLE]] : !tvm_ffi.c.function() : () -> !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[EMPTY_HANDLE]] : !tvm_ffi.c.function
// CHECK: return %[[EMPTY]] : !tvm_ffi.c.array
func.func @empty_tuple() -> !tvm_ffi.tuple<> {
  %result = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %result : (!tvm_ffi.tuple<>) -> ()
  return %result : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @get_array(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]] : i64 -> !tvm_ffi.int
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]] : !tvm_ffi.c.function(%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[GET]] : !tvm_ffi.c.function
// CHECK: return %[[ITEM]]
func.func @get_array(%array: !tvm_ffi.array<!tvm_ffi.int>, %index: i64) -> !tvm_ffi.int {
  %result = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func private @identity(!tvm_ffi.c.array) -> !tvm_ffi.c.array
func.func private @identity(!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int>

// CHECK-LABEL: func.func @regions(
// CHECK-SAME: %[[REGION_ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[REGION_TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[REGION_READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.c.array
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = call @identity(%[[REGION_TUPLE]]) : (!tvm_ffi.c.array) -> !tvm_ffi.c.array
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[REGION_READY]] -> (!tvm_ffi.c.array)
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi.c.array
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi.c.array
// CHECK: return %[[BRANCH]] : !tvm_ffi.c.array
func.func @regions(%array: !tvm_ffi.array<!tvm_ffi.int>, %tuple: !tvm_ffi.tuple<!tvm_ffi.int>, %ready: i1) -> !tvm_ffi.array<!tvm_ffi.int> {
  %called = func.call @identity(%tuple) : (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %selected = scf.if %ready -> (!tvm_ffi.array<!tvm_ffi.int>) {
    scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
  } else {
    scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
  }
  return %selected : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @size_array(
// CHECK-SAME: %[[SIZED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArraySize"
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SIZE]] : !tvm_ffi.c.function(%[[SIZED]]) : (!tvm_ffi.c.array) -> !tvm_ffi.int
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SIZE]] : !tvm_ffi.c.function
// CHECK: return %[[LENGTH]]
func.func @size_array(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.int {
  %result = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @tuple_ops(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO]]
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[NESTED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]] : !tvm_ffi.c.function(%[[TUPLE]], %[[OFFSET]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[GET]] : !tvm_ffi.c.function
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = arith.constant 1 : i64
// CHECK: %[[SECOND_OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ONE]]
// CHECK: %[[SECOND_GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SECOND_GET]] : !tvm_ffi.c.function(%[[TUPLE]], %[[SECOND_OFFSET]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND_GET]] : !tvm_ffi.c.function
// CHECK: return %[[NESTED]], %[[SECOND]] : !tvm_ffi.c.array, !tvm_ffi.int
func.func @tuple_ops(%tuple: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int) {
  %nested, %integer = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int)
  return %nested, %integer : !tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int
}
