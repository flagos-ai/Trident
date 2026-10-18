// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s
// CHECK-LABEL: tvm_ffi.func @results(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.monad.result<i64>
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]] -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.bind %[[INPUT]]#0, %[[INPUT]]#1
// CHECK: ^bb{{[0-9]+}}(%[[PAYLOAD:[a-zA-Z0-9_]+]]: i64, %[[UNIT_PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi.none):
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]] : i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: tvm_ffi.monad.yield %[[RESULT]]
// CHECK: tvm_ffi.return %[[OUTPUT]]
tvm_ffi.func @results(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<i64> {
  %input:2 = scf.if %ready -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
    %input_pure_none = tvm_ffi.constant.none
    %input_pure:2 = tvm_ffi.monad.pure %value, %input_pure_none : (i64, !tvm_ffi.none) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>)
    scf.yield %input_pure#0, %input_pure#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %input_fail:2 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %input_fail#0, %input_fail#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input#0, %input#1 : (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%payload: i64, %unit_payload: !tvm_ffi.none):
    %sum = arith.addi %payload, %value : i64
    %result = scf.if %ready -> (!tvm_ffi.monad.result<i64>) {
      %result_pure = tvm_ffi.monad.pure %sum : (i64) -> !tvm_ffi.monad.result<i64>
      scf.yield %result_pure : !tvm_ffi.monad.result<i64>
    } else {
      %result_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
      scf.yield %result_fail : !tvm_ffi.monad.result<i64>
    }
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<i64>
  }

  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}
