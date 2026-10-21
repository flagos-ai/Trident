// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --verify-each | FileCheck %s --check-prefix=IR
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s --check-prefix=LOWER --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// LOWER-LABEL:   func.func @empty() {
// LOWER:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// LOWER:           scf.if %[[CONSTANT_0]] {
// LOWER:             func.call @record() : () -> ()
// LOWER:           return
// LOWER-LABEL:   func.func @no_inputs(
// LOWER-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.int) -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// LOWER:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant true
// LOWER:           %[[IF_0:[a-zA-Z0-9_$.-]+]]:2 = scf.if %[[CONSTANT_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// LOWER-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// LOWER-DAG:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// LOWER:             %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER-DAG:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[ARG0]] : !tvm_ffi.int -> !tvm_ffi.any
// LOWER-DAG:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// LOWER:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER:             scf.yield %[[TUPLE_0]], %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER:           } else {
// LOWER:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// LOWER:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// LOWER:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// LOWER:             %[[TUPLE_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// LOWER:             %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// LOWER:             %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER:             scf.yield %[[TUPLE_2]], %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER:           return %[[VAL_0:[a-zA-Z0-9_$.-]+]]#0, %[[VAL_0]]#1 : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>, !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LOWER-LABEL:   func.func @no_outputs(
// LOWER-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>,
// LOWER-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// LOWER:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// LOWER:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_0]]#0 : !tvm_ffi.bool -> i1
// LOWER:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// LOWER:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_1]]#0 : !tvm_ffi.bool -> i1
// LOWER:           %[[ANDI_0:[a-zA-Z0-9_$.-]+]] = arith.andi %[[GET_0]], %[[GET_1]] : i1
// LOWER:           scf.if %[[ANDI_0]] {
// LOWER:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_0]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// LOWER:             %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_1]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// LOWER:             func.call @record_values(%[[DOWNCAST_0]], %[[DOWNCAST_1]]) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
// LOWER:           return

// IR-LABEL: func.func @empty(
// IR: tvm_ffi.monad.bind : () -> () {
// IR-NEXT: call @record()
// IR-NEXT: tvm_ffi.monad.yield
// LLVM-LABEL: llvm.func @empty(
// LLVM: llvm.call @record()
// LLVM: llvm.return
func.func @empty() {
  tvm_ffi.monad.bind : () -> () {
    func.call @record() : () -> ()
    tvm_ffi.monad.yield
  }
  return
}

// No inputs means the continuation executes unconditionally and may produce
// any number of Results.
// IR-LABEL: func.func @no_inputs(
// IR: tvm_ffi.monad.bind : () -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
// LLVM-LABEL: llvm.func @no_inputs(
// LLVM: llvm.return
func.func @no_inputs(%value: !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  %output:2 = tvm_ffi.monad.bind : () -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
    %pure:2 = tvm_ffi.monad.pure %value, %value : (!tvm_ffi.int, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>)
    tvm_ffi.monad.yield %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
}

// Every input must succeed before the side effect executes. The no-output
// failure path needs neither a failed Result nor an else branch.
// IR-LABEL: func.func @no_outputs(
// IR: tvm_ffi.monad.bind
// IR-SAME: -> () {
// LLVM-LABEL: llvm.func @no_outputs(
// LLVM: llvm.call @record_values
// LLVM: llvm.return
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
