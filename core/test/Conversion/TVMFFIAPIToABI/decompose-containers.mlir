// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-monad --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi_api.array<' --implicit-check-not='!tvm_ffi_api.tuple<' --implicit-check-not='tvm_ffi_monad.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.array,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.array,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: i64,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi_abi.array {
// CHECK: %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_0]] : !tvm_ffi_abi.function(%[[ARG0]], %[[TO_0]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.bool
// CHECK: %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_2]] : !tvm_ffi_abi.function(%[[ARG0]], %[[TO_1]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.any
// CHECK: %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.get %[[VAL_1]] : !tvm_ffi_abi.bool -> i1
// CHECK: %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi_abi.array) {
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.downcast %[[VAL_3]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.array
// CHECK: %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_3]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_4]] : !tvm_ffi_abi.function(%[[DOWNCAST_0]], %[[TO_2]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK: %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_4]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_6]] : !tvm_ffi_abi.function(%[[DOWNCAST_0]], %[[TO_3]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
// CHECK: %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG3]] -> (!tvm_ffi_abi.array) {
// CHECK: scf.yield %[[VAL_5]] : !tvm_ffi_abi.array
// CHECK: } else {
// CHECK: %[[VAL_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_8]] : !tvm_ffi_abi.function(%[[VAL_7]]) : (!tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK: scf.yield %[[VAL_9]] : !tvm_ffi_abi.array
// CHECK: %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK: %[[TO_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_6]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_10]] : !tvm_ffi_abi.function(%[[ARG1]], %[[TO_4]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.bool
// CHECK: %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK: %[[TO_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[CONSTANT_7]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_13:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_12]] : !tvm_ffi_abi.function(%[[ARG1]], %[[TO_5]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.any
// CHECK: %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.get %[[VAL_11]] : !tvm_ffi_abi.bool -> i1
// CHECK: %[[IF_2:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_1]] -> (!tvm_ffi_abi.array) {
// CHECK: %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.downcast %[[VAL_13]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.int
// CHECK: %[[TO_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.to %[[ARG2]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[VAL_14:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem" : !tvm_ffi_abi.function
// CHECK: %[[VAL_15:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_14]] : !tvm_ffi_abi.function(%[[IF_1]], %[[TO_6]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
// CHECK: %[[VAL_16:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_17:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_16]] : !tvm_ffi_abi.function(%[[VAL_15]], %[[DOWNCAST_1]], %[[VAL_7]]) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.upcast %[[VAL_17]] : !tvm_ffi_abi.array -> !tvm_ffi_abi.any
// CHECK-DAG: %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[VAL_18:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_19:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_18]] : !tvm_ffi_abi.function(%[[CONSTANT_8]], %[[UPCAST_0]]) : (!tvm_ffi_abi.bool, !tvm_ffi_abi.any) -> !tvm_ffi_abi.array
// CHECK: scf.yield %[[VAL_19]] : !tvm_ffi_abi.array
// CHECK: } else {
// CHECK: %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.upcast %[[CONSTANT_9]] : !tvm_ffi_abi.none -> !tvm_ffi_abi.any
// CHECK: %[[VAL_20:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_21:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_20]] : !tvm_ffi_abi.function(%[[CONSTANT_10]], %[[UPCAST_1]]) : (!tvm_ffi_abi.bool, !tvm_ffi_abi.any) -> !tvm_ffi_abi.array
// CHECK: scf.yield %[[VAL_21]] : !tvm_ffi_abi.array
// CHECK: scf.yield %[[IF_2]] : !tvm_ffi_abi.array
// CHECK: } else {
// CHECK: %[[CONSTANT_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[CONSTANT_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool false
// CHECK: %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.upcast %[[CONSTANT_11]] : !tvm_ffi_abi.none -> !tvm_ffi_abi.any
// CHECK: %[[VAL_22:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_23:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_22]] : !tvm_ffi_abi.function(%[[CONSTANT_12]], %[[UPCAST_2]]) : (!tvm_ffi_abi.bool, !tvm_ffi_abi.any) -> !tvm_ffi_abi.array
// CHECK: scf.yield %[[VAL_23]] : !tvm_ffi_abi.array
// CHECK: return %[[IF_0]] : !tvm_ffi_abi.array

// Nested continuations combine remapped block arguments, lexical captures and
// structural SCF conversion. No semantic container cast may escape conversion.
func.func @nested_containers(%input: !tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>>, %scalar: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %index: i64, %predicate: i1) -> !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>> {
  %outer = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>>) -> !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>> {
  ^bb0(%tuple: !tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>):
    %array, %capture = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>) -> (!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int)
    %chosen = scf.if %predicate -> (!tvm_ffi_api.array<!tvm_ffi_abi.int>) {
      scf.yield %array : !tvm_ffi_api.array<!tvm_ffi_abi.int>
    } else {
      %fallback = tvm_ffi_api.array.create(%capture) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
      scf.yield %fallback : !tvm_ffi_api.array<!tvm_ffi_abi.int>
    }
    %inner = tvm_ffi_monad.bind %scalar : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>> {
    ^bb1(%value: !tvm_ffi_abi.int):
      %item = tvm_ffi_api.array.get %chosen[%index] : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
      %result = tvm_ffi_api.array.create(%item, %value, %capture) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
      %pure = tvm_ffi_monad.pure %result : (!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>>
      tvm_ffi_monad.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>>
    }
    tvm_ffi_monad.yield %inner : !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>>
  }
  return %outer : !tvm_ffi_monad.result<!tvm_ffi_api.array<!tvm_ffi_abi.int>>
}

// -----

// CHECK-LABEL: func.func @casts(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.array, %[[INPUT:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.any) -> !tvm_ffi_abi.array {
// CHECK: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.upcast %[[ARG0]] : !tvm_ffi_abi.array -> !tvm_ffi_abi.any {test.keep = "wide"}
// CHECK: %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.downcast %[[INPUT]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.array {test.types = [!tvm_ffi_abi.array]}
// CHECK: return %[[DOWNCAST_0]] : !tvm_ffi_abi.array
// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.array) -> !tvm_ffi_abi.array {
// CHECK-DAG: %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.upcast %[[ARG0]] : !tvm_ffi_abi.array -> !tvm_ffi_abi.any
// CHECK-DAG: %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.constant.bool true
// CHECK: %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi_abi.function.call %[[VAL_0]] : !tvm_ffi_abi.function(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi_abi.bool, !tvm_ffi_abi.any) -> !tvm_ffi_abi.array
// CHECK: tvm_ffi_abi.object.dec_ref %[[VAL_0]] : !tvm_ffi_abi.function
// CHECK: return %[[VAL_1]] : !tvm_ffi_abi.array
// CHECK-LABEL: func.func @opaque_passthrough(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.any,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.exception,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.object,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_$.-]+]]: !tvm_ffi_abi.any) -> (!tvm_ffi_abi.any, !tvm_ffi_abi.exception, !tvm_ffi_abi.object, !tvm_ffi_abi.any) {
// CHECK: return %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !tvm_ffi_abi.any, !tvm_ffi_abi.exception, !tvm_ffi_abi.object, !tvm_ffi_abi.any

func.func @casts(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>, %input: !tvm_ffi_api.any) -> !tvm_ffi_api.array<!tvm_ffi_abi.int> {
  %wide = tvm_ffi_api.upcast %array : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_api.any {test.keep = "wide"}
  %narrow = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_api.array<!tvm_ffi_abi.int> {test.types = [!tvm_ffi_api.tuple<!tvm_ffi_abi.int>]}
  return %narrow : !tvm_ffi_api.array<!tvm_ffi_abi.int>
}

func.func @nested_containers(%nested: !tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>>) -> !tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>>> {
  %result = tvm_ffi_monad.pure %nested {test.types = [!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>]} : (!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>>) -> !tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>>>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>>>>
}

func.func @opaque_passthrough(%any: !tvm_ffi_api.any, %exception: !tvm_ffi_abi.exception, %object: !tvm_ffi_abi.object, %union: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) -> (!tvm_ffi_api.any, !tvm_ffi_abi.exception, !tvm_ffi_abi.object, !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) {
  return %any, %exception, %object, %union : !tvm_ffi_api.any, !tvm_ffi_abi.exception, !tvm_ffi_abi.object, !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
}

// -----

// CHECK-LABEL: func.func @container_attributes(
// CHECK-SAME: attributes {test.types = [!tvm_ffi_abi.array, {nested = !tvm_ffi_abi.array}]}
func.func @container_attributes() attributes {test.types = [!tvm_ffi_api.array<!tvm_ffi_abi.int>, {nested = !tvm_ffi_api.tuple<!tvm_ffi_abi.int>}]} {
  return
}

// CHECK-LABEL: func.func @create_array(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.Array"
// CHECK: %[[CALLED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[HANDLE]] : !tvm_ffi_abi.function(%[[VALUE]]) : (!tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[HANDLE]] : !tvm_ffi_abi.function
// CHECK: return %[[CALLED]]
func.func @create_array(%value: !tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int> {
  %result = tvm_ffi_api.array.create(%value) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  return %result : !tvm_ffi_api.array<!tvm_ffi_abi.int>
}

// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.Array"
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[EMPTY_HANDLE]] : !tvm_ffi_abi.function() : () -> !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[EMPTY_HANDLE]] : !tvm_ffi_abi.function
// CHECK: return %[[EMPTY]] : !tvm_ffi_abi.array
func.func @empty_tuple() -> !tvm_ffi_api.tuple<> {
  %result = tvm_ffi_api.tuple.construct() : () -> !tvm_ffi_api.tuple<>
  tvm_ffi_api.tuple.unconstruct %result : (!tvm_ffi_api.tuple<>) -> ()
  return %result : !tvm_ffi_api.tuple<>
}

// CHECK-LABEL: func.func @get_array(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[INDEX]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem"
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[GET]] : !tvm_ffi_abi.function(%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[GET]] : !tvm_ffi_abi.function
// CHECK: return %[[ITEM]]
func.func @get_array(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>, %index: i64) -> !tvm_ffi_abi.int {
  %result = tvm_ffi_api.array.get %array[%index] : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// CHECK-LABEL: func.func private @identity(!tvm_ffi_abi.array) -> !tvm_ffi_abi.array
func.func private @identity(!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>

// CHECK-LABEL: func.func @regions(
// CHECK-SAME: %[[REGION_ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array,
// CHECK-SAME: %[[REGION_TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array,
// CHECK-SAME: %[[REGION_READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi_abi.array
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = call @identity(%[[REGION_TUPLE]]) : (!tvm_ffi_abi.array) -> !tvm_ffi_abi.array
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[REGION_READY]] -> (!tvm_ffi_abi.array)
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi_abi.array
// CHECK: scf.yield %[[REGION_ARRAY]] : !tvm_ffi_abi.array
// CHECK: return %[[BRANCH]] : !tvm_ffi_abi.array
func.func @regions(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>, %tuple: !tvm_ffi_api.tuple<!tvm_ffi_abi.int>, %ready: i1) -> !tvm_ffi_api.array<!tvm_ffi_abi.int> {
  %called = func.call @identity(%tuple) : (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
  %selected = scf.if %ready -> (!tvm_ffi_api.array<!tvm_ffi_abi.int>) {
    scf.yield %array : !tvm_ffi_api.array<!tvm_ffi_abi.int>
  } else {
    scf.yield %array : !tvm_ffi_api.array<!tvm_ffi_abi.int>
  }
  return %selected : !tvm_ffi_api.array<!tvm_ffi_abi.int>
}

// CHECK-LABEL: func.func @size_array(
// CHECK-SAME: %[[SIZED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array)
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArraySize"
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[SIZE]] : !tvm_ffi_abi.function(%[[SIZED]]) : (!tvm_ffi_abi.array) -> !tvm_ffi_abi.int
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[SIZE]] : !tvm_ffi_abi.function
// CHECK: return %[[LENGTH]]
func.func @size_array(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int {
  %result = tvm_ffi_api.array.size %array : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// CHECK-LABEL: func.func @tuple_ops(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array)
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[ZERO]]
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem"
// CHECK: %[[NESTED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[GET]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[OFFSET]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[GET]] : !tvm_ffi_abi.function
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = arith.constant 1 : i64
// CHECK: %[[SECOND_OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[ONE]]
// CHECK: %[[SECOND_GET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[SECOND_GET]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[SECOND_OFFSET]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[SECOND_GET]] : !tvm_ffi_abi.function
// CHECK: return %[[NESTED]], %[[SECOND]] : !tvm_ffi_abi.array, !tvm_ffi_abi.int
func.func @tuple_ops(%tuple: !tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>) -> (!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int) {
  %nested, %integer = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int>) -> (!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int)
  return %nested, %integer : !tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.int
}
