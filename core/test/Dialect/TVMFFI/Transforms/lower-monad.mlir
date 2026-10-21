// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.transfer'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=FULL --implicit-check-not='!tvm_ffi.'

// CHECK-LABEL:   func.func @construct(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
// CHECK-DAG:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func @construct_multiple(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.none,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.function) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-DAG:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG1]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK-DAG:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-DAG:           %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG2]] : !tvm_ffi.c.function -> !tvm_ffi.any
// CHECK-DAG:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func @failures() -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[TUPLE_0]], %[[TUPLE_1]], %[[TUPLE_2]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func @forward(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = call @identity(%[[ARG0]]) : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           return %[[VAL_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-LABEL:   func.func @unpack_any(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_float(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_function(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_int(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_none(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_tensor(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any
// CHECK-LABEL:   func.func @unpack_union(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// CHECK:           return %[[GET_0]], %[[TUPLE_0]]#1 : i1, !tvm_ffi.any

func.func @construct(%value: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %result = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
}

func.func @construct_multiple(%value: !tvm_ffi.int, %none: !tvm_ffi.none, %function: !tvm_ffi.c.function) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>) {
  %results:3 = tvm_ffi.monad.pure %value, %none, %function : (!tvm_ffi.int, !tvm_ffi.none, !tvm_ffi.c.function) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>)
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

func.func @failures() -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>) {
  %results:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

func.func @forward(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %result = call @identity(%input) : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
}

func.func private @identity(!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>

func.func @unpack_any(%result: !tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_float(%result: !tvm_ffi.monad.result<!tvm_ffi.float>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.float>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_function(%result: !tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_int(%result: !tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_none(%result: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_tensor(%result: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

func.func @unpack_union(%result: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// FULL-LABEL: llvm.func @construct(
// FULL-LABEL: llvm.func @construct_multiple(
// FULL-LABEL: llvm.func @failures(
// FULL-LABEL: llvm.func @forward(
// FULL-LABEL: llvm.func @unpack_any(
// FULL-LABEL: llvm.func @unpack_float(
// FULL-LABEL: llvm.func @unpack_function(
// FULL-LABEL: llvm.func @unpack_int(
// FULL-LABEL: llvm.func @unpack_none(
// FULL-LABEL: llvm.func @unpack_tensor(
// FULL-LABEL: llvm.func @unpack_union(
