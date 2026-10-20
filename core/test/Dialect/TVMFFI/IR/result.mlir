// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s
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
