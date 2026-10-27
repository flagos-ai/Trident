// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-each | trident-core-opt --split-input-file --verify-each | FileCheck %s

tvm_ffi.func @callee(!tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor
// CHECK: %[[MONAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.call @callee(%[[INPUT]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @callee(%[[INPUT]]) : (!tvm_ffi.c.tensor) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK: return %[[MONAD]], %[[OK]], %[[VALUE]]
func.func @invoke(%input: !tvm_ffi.c.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.bool, !tvm_ffi.any) {
  %monad = tvm_ffi.monad.call @callee(%input) : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.c.tensor) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %monad, %ok, %value : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>, !tvm_ffi.bool, !tvm_ffi.any
}

// -----

// CHECK-LABEL: func.func @results(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.monad.result<!tvm_ffi.int>
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]] -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.bind %[[INPUT]]#0, %[[INPUT]]#1
// CHECK: ^bb{{[0-9]+}}(%[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[UNIT_PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi.none):
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]] : !tvm_ffi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]] : i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: tvm_ffi.monad.yield %[[RESULT]]
// CHECK: return %[[OUTPUT]]
func.func @results(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %input:2 = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
    %input_pure_none = tvm_ffi.constant.none
    %value_boxed = tvm_ffi.to %value : i64 -> !tvm_ffi.int
    %input_pure:2 = tvm_ffi.monad.pure %value_boxed, %input_pure_none : (!tvm_ffi.int, !tvm_ffi.none) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>)
    scf.yield %input_pure#0, %input_pure#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %input_fail:2 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %input_fail#0, %input_fail#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input#0, %input#1 : (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%payload_boxed: !tvm_ffi.int, %unit_payload: !tvm_ffi.none):
    %payload = tvm_ffi.get %payload_boxed : !tvm_ffi.int -> i64
    %sum = arith.addi %payload, %value : i64
    %result = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.int>) {
      %sum_boxed = tvm_ffi.to %sum : i64 -> !tvm_ffi.int
      %result_pure = tvm_ffi.monad.pure %sum_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
      scf.yield %result_pure : !tvm_ffi.monad.result<!tvm_ffi.int>
    } else {
      %result_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>
      scf.yield %result_fail : !tvm_ffi.monad.result<!tvm_ffi.int>
    }
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<!tvm_ffi.int>
  }

  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// CHECK-LABEL: tvm_ffi.func @tuple(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[FIRST]], %[[SECOND]])
// CHECK-DAG: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TUPLE]]
// CHECK-DAG: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: tvm_ffi.return %[[OK]], %[[PAYLOAD]] : !tvm_ffi.bool, !tvm_ffi.any
tvm_ffi.func @tuple(%first: !tvm_ffi.int, %second: !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.c.tensor> {
  %tuple = tvm_ffi.tuple.construct(%first, %second) : (!tvm_ffi.int, !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.c.tensor>
  %payload = tvm_ffi.upcast %tuple : !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.c.tensor> -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// CHECK-LABEL: tvm_ffi.func @failure(
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool false
// CHECK: tvm_ffi.return %[[OK]], %[[PAYLOAD]]
tvm_ffi.func @failure() -> !tvm_ffi.c.tensor {
  %none = tvm_ffi.constant.none
  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool false
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// -----

// Representatives for Any identity, Base Any compatibility, native function
// handles and POD. Cast compatibility owns the full
// type matrix; unpack owns inference of the success and widened payload results.

// CHECK-LABEL: func.func @any(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.any>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi.any
func.func @any(%input: !tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// CHECK-LABEL: func.func @c_any(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.any>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.c.any>) -> (i1, !tvm_ffi.c.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi.c.any
func.func @c_any(%input: !tvm_ffi.monad.result<!tvm_ffi.c.any>) -> (i1, !tvm_ffi.c.any) {
  %ok, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.c.any>) -> (i1, !tvm_ffi.c.any)
  return %ok, %value : i1, !tvm_ffi.c.any
}

// CHECK-LABEL: func.func @function(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.function>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi.any
func.func @function(%input: !tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi.any
func.func @scalar(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}
