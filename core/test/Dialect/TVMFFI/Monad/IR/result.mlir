// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-each | trident-core-opt --split-input-file --verify-each | FileCheck %s

tvm_ffi_api.func @callee(!tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor
// CHECK: %[[MONAD:[a-zA-Z0-9_]+]] = tvm_ffi_monad.call @callee(%[[INPUT]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_api.call @callee(%[[INPUT]]) : (!tvm_ffi_abi.tensor) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
// CHECK: return %[[MONAD]], %[[OK]], %[[VALUE]]
func.func @invoke(%input: !tvm_ffi_abi.tensor) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_abi.bool, !tvm_ffi_api.any) {
  %monad = tvm_ffi_monad.call @callee(%input) : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  %ok, %value = tvm_ffi_api.call @callee(%input) : (!tvm_ffi_abi.tensor) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return %monad, %ok, %value : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>, !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// -----

// CHECK-LABEL: func.func @results(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]] -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi_monad.bind %[[INPUT]]#0, %[[INPUT]]#1
// CHECK: ^bb{{[0-9]+}}(%[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[UNIT_PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.none):
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_api.get %[[BOXED]] : !tvm_ffi_abi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]] : i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: tvm_ffi_monad.yield %[[RESULT]]
// CHECK: return %[[OUTPUT]]
func.func @results(%value: i64, %ready: i1) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  %input:2 = scf.if %ready -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
    %input_pure_none = tvm_ffi_abi.constant.none
    %value_boxed = tvm_ffi_abi.to %value : i64 -> !tvm_ffi_abi.int
    %input_pure_0 = tvm_ffi_monad.pure %value_boxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    %input_pure_1 = tvm_ffi_monad.pure %input_pure_none : (!tvm_ffi_abi.none) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    scf.yield %input_pure_0, %input_pure_1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  } else {
    %input_fail:2 = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    scf.yield %input_fail#0, %input_fail#1 : !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  }
  %output = tvm_ffi_monad.bind %input#0, %input#1 : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%payload_boxed: !tvm_ffi_abi.int, %unit_payload: !tvm_ffi_abi.none):
    %payload = tvm_ffi_api.get %payload_boxed : !tvm_ffi_abi.int -> i64
    %sum = arith.addi %payload, %value : i64
    %result = scf.if %ready -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
      %sum_boxed = tvm_ffi_abi.to %sum : i64 -> !tvm_ffi_abi.int
      %result_pure = tvm_ffi_monad.pure %sum_boxed : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
      scf.yield %result_pure : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    } else {
      %result_fail = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
      scf.yield %result_fail : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    }
    tvm_ffi_monad.yield %result : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }

  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// -----

// CHECK-LABEL: tvm_ffi_api.func @tuple(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[FIRST]], %[[SECOND]])
// CHECK-DAG: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[TUPLE]]
// CHECK-DAG: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.bool true
// CHECK: tvm_ffi_api.return %[[OK]], %[[PAYLOAD]] : !tvm_ffi_abi.bool, !tvm_ffi_api.any
tvm_ffi_api.func @tuple(%first: !tvm_ffi_abi.int, %second: !tvm_ffi_abi.tensor) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.tensor> {
  %tuple = tvm_ffi_api.tuple.construct(%first, %second) : (!tvm_ffi_abi.int, !tvm_ffi_abi.tensor) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.tensor>
  %payload = tvm_ffi_api.upcast %tuple : !tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.tensor> -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// CHECK-LABEL: tvm_ffi_api.func @failure(
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[NONE]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.bool false
// CHECK: tvm_ffi_api.return %[[OK]], %[[PAYLOAD]]
tvm_ffi_api.func @failure() -> !tvm_ffi_abi.tensor {
  %none = tvm_ffi_abi.constant.none
  %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool false
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// -----

// Representatives for Any identity, Base Any compatibility, native function
// handles and POD. Cast compatibility owns the full
// type matrix; unpack owns inference of the success and widened payload results.

// CHECK-LABEL: func.func @any(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_api.any>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_monad.unpack %[[INPUT]] : (!tvm_ffi_monad.result<!tvm_ffi_api.any>) -> (i1, !tvm_ffi_api.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi_api.any
func.func @any(%input: !tvm_ffi_monad.result<!tvm_ffi_api.any>) -> (i1, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_api.any>) -> (i1, !tvm_ffi_api.any)
  return %ok, %value : i1, !tvm_ffi_api.any
}

// CHECK-LABEL: func.func @c_any(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.any>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_monad.unpack %[[INPUT]] : (!tvm_ffi_monad.result<!tvm_ffi_abi.any>) -> (i1, !tvm_ffi_abi.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi_abi.any
func.func @c_any(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.any>) -> (i1, !tvm_ffi_abi.any) {
  %ok, %value = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.any>) -> (i1, !tvm_ffi_abi.any)
  return %ok, %value : i1, !tvm_ffi_abi.any
}

// CHECK-LABEL: func.func @function(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.function>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_monad.unpack %[[INPUT]] : (!tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> (i1, !tvm_ffi_api.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi_api.any
func.func @function(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> (i1, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> (i1, !tvm_ffi_api.any)
  return %ok, %value : i1, !tvm_ffi_api.any
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.int>)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_monad.unpack %[[INPUT]] : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (i1, !tvm_ffi_api.any)
// CHECK-NEXT: return %[[OK]], %[[VALUE]] : i1, !tvm_ffi_api.any
func.func @scalar(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (i1, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (i1, !tvm_ffi_api.any)
  return %ok, %value : i1, !tvm_ffi_api.any
}
