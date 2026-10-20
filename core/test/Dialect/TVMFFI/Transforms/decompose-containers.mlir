// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.array.' --implicit-check-not='tvm_ffi.tuple.' --implicit-check-not='tvm_ffi.monad.'
// RUN: trident-core-opt %s | trident-core-opt --lower-tvm-ffi-monad --decompose-tvm-ffi | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.array.' --implicit-check-not='tvm_ffi.tuple.' --implicit-check-not='tvm_ffi.monad.'

// CHECK-LABEL: func.func @container_attributes(
// CHECK-SAME: attributes {test.types = [!tvm_ffi.c.array, {nested = !tvm_ffi.c.array}]}
func.func @container_attributes() attributes {test.types = [!tvm_ffi.array<!tvm_ffi.int>, {nested = !tvm_ffi.tuple<!tvm_ffi.int>}]} {
  return
}

// CHECK-LABEL: func.func @create_array(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[CALLED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[VALUE]]) {tvm_ffi.cleanup} : (!tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: return %[[CALLED]]
func.func @create_array(%value: !tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int> {
  %result = tvm_ffi.array.create(%value) {tvm_ffi.cleanup} : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
  return %result : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @create_exception(
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi.exception)
// CHECK: %[[EXCEPTION_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[EXCEPTION_ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[EXCEPTION_HANDLE]](%[[EXCEPTION]]) {tvm_ffi.cleanup} : (!tvm_ffi.exception) -> !tvm_ffi.c.array
// CHECK: return %[[EXCEPTION_ARRAY]]
func.func @create_exception(%exception: !tvm_ffi.exception) -> !tvm_ffi.array<!tvm_ffi.exception> {
  %result = tvm_ffi.array.create(%exception) {tvm_ffi.cleanup} : (!tvm_ffi.exception) -> !tvm_ffi.array<!tvm_ffi.exception>
  return %result : !tvm_ffi.array<!tvm_ffi.exception>
}
// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[EMPTY_HANDLE]]() {tvm_ffi.cleanup} : () -> !tvm_ffi.c.array
// CHECK: return %[[EMPTY]] : !tvm_ffi.c.array
func.func @empty_tuple() -> !tvm_ffi.tuple<> {
  %result = tvm_ffi.tuple.construct() {tvm_ffi.cleanup} : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %result {tvm_ffi.cleanup} : (!tvm_ffi.tuple<>) -> ()
  return %result : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @get_array(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]] : i64 -> !tvm_ffi.int
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]](%[[ARRAY]], %[[BOXED]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: return %[[ITEM]]
func.func @get_array(%array: !tvm_ffi.array<!tvm_ffi.int>, %index: i64) -> !tvm_ffi.int {
  %result = tvm_ffi.array.get %array[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func private @identity(!tvm_ffi.c.array) -> !tvm_ffi.c.array
func.func private @identity(!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int>

// CHECK-LABEL: func.func @loop(
// CHECK-SAME: %[[INITIAL:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[UPPER:[a-zA-Z0-9_]+]]: index)
// CHECK: %[[LOWER:[a-zA-Z0-9_]+]] = arith.constant 0 : index
// CHECK: %[[STEP:[a-zA-Z0-9_]+]] = arith.constant 1 : index
// CHECK: %[[LOOP:[a-zA-Z0-9_]+]] = scf.for %[[IV:[a-zA-Z0-9_]+]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] iter_args(%[[CURRENT:[a-zA-Z0-9_]+]] = %[[INITIAL]]) -> (!tvm_ffi.c.array)
// CHECK: scf.yield %[[CURRENT]] : !tvm_ffi.c.array
// CHECK: return %[[LOOP]] : !tvm_ffi.c.array
func.func @loop(%initial: !tvm_ffi.array<!tvm_ffi.int>, %upper: index) -> !tvm_ffi.array<!tvm_ffi.int> {
  %lower = arith.constant 0 : index
  %step = arith.constant 1 : index
  %result = scf.for %iv = %lower to %upper step %step iter_args(%current = %initial) -> (!tvm_ffi.array<!tvm_ffi.int>) {
    scf.yield %current : !tvm_ffi.array<!tvm_ffi.int>
  }
  return %result : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @regions(
// CHECK-SAME: %[[REGION_ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[REGION_TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[REGION_READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.c.array
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = call @identity(%[[REGION_TUPLE]]) : (!tvm_ffi.c.array) -> !tvm_ffi.c.array
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[REGION_READY]] -> (!tvm_ffi.c.array)
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi.c.array
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi.c.array
// CHECK: cf.br ^bb1(%[[BRANCH]] : !tvm_ffi.c.array)
// CHECK: ^bb1(%[[BLOCK:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array):
// CHECK: return %[[BLOCK]] : !tvm_ffi.c.array
func.func @regions(%array: !tvm_ffi.array<!tvm_ffi.int>, %tuple: !tvm_ffi.tuple<!tvm_ffi.int>, %ready: i1) -> !tvm_ffi.array<!tvm_ffi.int> {
  %called = func.call @identity(%tuple) : (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %selected = scf.if %ready -> (!tvm_ffi.array<!tvm_ffi.int>) {
    scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
  } else {
    scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
  }
  cf.br ^next(%selected : !tvm_ffi.array<!tvm_ffi.int>)
^next(%value: !tvm_ffi.array<!tvm_ffi.int>):
  return %value : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @size_array(
// CHECK-SAME: %[[SIZED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArraySize"
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SIZE]](%[[SIZED]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array) -> !tvm_ffi.int
// CHECK: return %[[LENGTH]]
func.func @size_array(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.int {
  %result = tvm_ffi.array.size %array {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// CHECK-LABEL: func.func @tuple_ops(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO]]
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[NESTED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]](%[[TUPLE]], %[[OFFSET]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = arith.constant 1 : i64
// CHECK: %[[SECOND_OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ONE]]
// CHECK: %[[SECOND_GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SECOND_GET]](%[[TUPLE]], %[[SECOND_OFFSET]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: return %[[NESTED]], %[[SECOND]] : !tvm_ffi.c.array, !tvm_ffi.int
func.func @tuple_ops(%tuple: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int) {
  %nested, %integer = tvm_ffi.tuple.unconstruct %tuple {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int)
  return %nested, %integer : !tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int
}
