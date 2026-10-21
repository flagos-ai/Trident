// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.array.' --implicit-check-not='tvm_ffi.tuple.' --implicit-check-not='tvm_ffi.monad.'
// RUN: trident-core-opt %s | trident-core-opt --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.array.' --implicit-check-not='tvm_ffi.tuple.' --implicit-check-not='tvm_ffi.monad.'

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

// CHECK-LABEL: func.func @create_exception(
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.exception)
// CHECK: %[[EXCEPTION_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[EXCEPTION_ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[EXCEPTION_HANDLE]] : !tvm_ffi.c.function(%[[EXCEPTION]]) : (!tvm_ffi.c.exception) -> !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[EXCEPTION_HANDLE]] : !tvm_ffi.c.function
// CHECK: return %[[EXCEPTION_ARRAY]]
func.func @create_exception(%exception: !tvm_ffi.c.exception) -> !tvm_ffi.array<!tvm_ffi.c.exception> {
  %result = tvm_ffi.array.create(%exception) : (!tvm_ffi.c.exception) -> !tvm_ffi.array<!tvm_ffi.c.exception>
  return %result : !tvm_ffi.array<!tvm_ffi.c.exception>
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
