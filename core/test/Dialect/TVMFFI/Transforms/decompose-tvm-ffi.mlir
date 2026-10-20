// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s --implicit-check-not=tvm_ffi.monad.bind --implicit-check-not=tvm_ffi.monad.yield

// CHECK-LABEL:   func.func @aggregate(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i64,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG1]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[ARG0]] : i64 -> !tvm_ffi.int
// CHECK:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TO_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:             scf.yield %[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_1]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
// CHECK:             tvm_ffi.ObjectIncRef %[[UPCAST_2]] : !tvm_ffi.any
// CHECK:             scf.yield %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_0]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK:             %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[DOWNCAST_0]] : !tvm_ffi.int -> i64
// CHECK:             %[[ADDI_0:[a-zA-Z0-9_$.-]+]] = arith.addi %[[GET_1]], %[[ARG0]] : i64
// CHECK:             %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[ADDI_0]] : i64 -> !tvm_ffi.int
// CHECK:             %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TO_1]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_3]] : !tvm_ffi.any
// CHECK:             scf.yield %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_4]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield {tvm_ffi.retain} %[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[IF_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func private @helper(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: i64) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG1]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_1]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[ANDI_0:[a-zA-Z0-9_$.-]+]] = arith.andi %[[GET_0]], %[[GET_1]] : i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[ANDI_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_0]]#1 : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK:             %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_1]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK:             %[[GET_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[DOWNCAST_1]] : !tvm_ffi.int -> i64
// CHECK:             %[[ADDI_0:[a-zA-Z0-9_$.-]+]] = arith.addi %[[GET_2]], %[[ARG2]] : i64
// CHECK:             %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[ADDI_0]] : i64 -> !tvm_ffi.int
// CHECK:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TO_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_1]] : !tvm_ffi.any
// CHECK:             scf.yield %[[TUPLE_2]], %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield {tvm_ffi.retain} %[[TUPLE_4]], %[[TUPLE_5]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[VAL_0:[a-zA-Z0-9_$.-]+]]#0, %[[VAL_0]]#1 : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func @nested_mixed(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[GET_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_0]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK:             %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[DOWNCAST_0]] : !tvm_ffi.int -> i64
// CHECK:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG1]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:             %[[GET_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_1]]#0 : !tvm_ffi.bool -> i1
// CHECK:             %[[IF_1:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[GET_2]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:               %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_1]]#1 : !tvm_ffi.any -> !tvm_ffi.none
// CHECK:               %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:               %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_1]] : i64 -> !tvm_ffi.int
// CHECK:               %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TO_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK:               %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:               %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:               %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_2]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:               %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:               %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               tvm_ffi.ObjectDecRef %[[UPCAST_1]] : !tvm_ffi.any
// CHECK:               scf.yield %[[TUPLE_2]], %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             } else {
// CHECK:               %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:               %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:               %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:               %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_2]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:               %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:               %[[TUPLE_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_7]], %[[UPCAST_3]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               scf.yield {tvm_ffi.retain} %[[TUPLE_4]], %[[TUPLE_5]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[VAL_0:[a-zA-Z0-9_$.-]+]]#0, %[[VAL_0]]#1 : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_8]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_9]], %[[UPCAST_4]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_8]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_10]], %[[UPCAST_5]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield {tvm_ffi.retain} %[[TUPLE_6]], %[[TUPLE_7]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[VAL_1:[a-zA-Z0-9_$.-]+]]#0, %[[VAL_1]]#1 : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>






func.func @aggregate(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.int>) {
    %value_boxed = tvm_ffi.to %value : i64 -> !tvm_ffi.int
    %success = tvm_ffi.monad.pure %value_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield %success : !tvm_ffi.monad.result<!tvm_ffi.int>
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

func.func private @external(!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>

// Decomposition is independent of function ABI lowering and preserves captures.
func.func private @helper(%tensor: !tvm_ffi.monad.result<!tvm_ffi.tensor>, %scalar: !tvm_ffi.monad.result<!tvm_ffi.int>, %capture: i64) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  %output:2 = tvm_ffi.monad.bind %tensor, %scalar : (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  ^bb0(%payload: !tvm_ffi.tensor, %value_boxed: !tvm_ffi.int):
    %value = tvm_ffi.get %value_boxed : !tvm_ffi.int -> i64
    %sum = arith.addi %value, %capture : i64
    %sum_boxed = tvm_ffi.to %sum : i64 -> !tvm_ffi.int
    %pure:2 = tvm_ffi.monad.pure %payload, %sum_boxed : (!tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>)
    tvm_ffi.monad.yield %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>
}

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
