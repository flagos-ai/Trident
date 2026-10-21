// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield'

// CHECK-LABEL:   func.func @bind_container(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.c.function
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.c.function
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]] : !tvm_ffi.c.function(%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.c.function
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.c.any -> !tvm_ffi.c.array
// CHECK-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]] : !tvm_ffi.c.function(%[[CONSTANT_3]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.c.function
// CHECK:             scf.yield %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.c.any
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]] : !tvm_ffi.c.function(%[[CONSTANT_5]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_6]] : !tvm_ffi.c.function
// CHECK:             scf.yield %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @casts(
// CHECK-SAME:                     %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.c.array -> !tvm_ffi.c.any {test.keep = "wide"}
// CHECK:           %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[UPCAST_0]] : !tvm_ffi.c.any -> !tvm_ffi.c.array {test.types = [!tvm_ffi.c.array]}
// CHECK:           return %[[DOWNCAST_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @nested_containers(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK-DAG:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.c.array -> !tvm_ffi.c.any
// CHECK-DAG:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]] : !tvm_ffi.c.function(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.c.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.c.function
// CHECK:           return %[[VAL_1]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @opaque_passthrough(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.any,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.exception,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.object,
// CHECK-SAME:      %[[ARG3:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.any) -> (!tvm_ffi.c.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.c.any) {
// CHECK:           return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !tvm_ffi.c.any, !tvm_ffi.c.exception, !tvm_ffi.object, !tvm_ffi.c.any

func.func @bind_container(%input: !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  ^bb0(%payload: !tvm_ffi.array<!tvm_ffi.int>):
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
  }
  return %result : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
}

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
