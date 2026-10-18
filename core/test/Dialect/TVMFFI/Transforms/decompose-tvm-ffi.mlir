// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not=tvm_ffi.monad.bind --implicit-check-not=tvm_ffi.monad.yield

// CHECK-LABEL: tvm_ffi.func @aggregate(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64,
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[PAYLOAD]], %[[VALUE]]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[SUM]]
// CHECK: scf.yield %[[RESULT]]
// CHECK: } else {
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[FAILED]]
// CHECK: tvm_ffi.return %[[OUTPUT]]
tvm_ffi.func @aggregate(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<i64> {
  %input = scf.if %ready -> (!tvm_ffi.monad.result<i64>) {
    %success = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    scf.yield %success : !tvm_ffi.monad.result<i64>
  } else {
    %failure = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
    scf.yield %failure : !tvm_ffi.monad.result<i64>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%payload: i64):
    %sum = arith.addi %payload, %value : i64
    %result = tvm_ffi.monad.pure %sum : (i64) -> !tvm_ffi.monad.result<i64>
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<i64>
  }

  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}

// CHECK-LABEL: tvm_ffi.func @external
tvm_ffi.func @external(!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64>

// Decomposition is independent of function ABI lowering and preserves captures.
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[TENSOR_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.tensor>, %[[SCALAR_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<i64>, %[[CAPTURE:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[TENSOR_OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[TENSOR_INPUT]]
// CHECK: %[[SCALAR_OK:[a-zA-Z0-9_]+]], %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[SCALAR_INPUT]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_OK]], %[[SCALAR_OK]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]]
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[SCALAR]], %[[CAPTURE]]
// CHECK: %[[PURE:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[TENSOR]], %[[SUM]]
// CHECK: scf.yield %[[PURE]]#0, %[[PURE]]#1
// CHECK: } else {
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[FAILED]]#0, %[[FAILED]]#1
// CHECK: return %[[OUTPUT]]#0, %[[OUTPUT]]#1
func.func private @helper(%tensor: !tvm_ffi.monad.result<!tvm_ffi.tensor>, %scalar: !tvm_ffi.monad.result<i64>, %capture: i64) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>) {
  %output:2 = tvm_ffi.monad.bind %tensor, %scalar : (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>) {
  ^bb0(%payload: !tvm_ffi.tensor, %value: i64):
    %sum = arith.addi %value, %capture : i64
    %pure:2 = tvm_ffi.monad.pure %payload, %sum : (!tvm_ffi.tensor, i64) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>)
    tvm_ffi.monad.yield %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<i64>
}

// CHECK-LABEL: tvm_ffi.func @nested_mixed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<i64>, %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if %[[OK]]
// CHECK: %[[UNIT_OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[UNIT]]
// CHECK: %[[INNER:[a-zA-Z0-9_]+]]:2 = scf.if %[[UNIT_OK]]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: %[[UNIT_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[PURE:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[PAYLOAD]], %[[UNIT_VALUE]]
// CHECK: scf.yield %[[PURE]]#0, %[[PURE]]#1
// CHECK: } else {
// CHECK: %[[INNER_FAIL:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[INNER_FAIL]]#0, %[[INNER_FAIL]]#1
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1
// CHECK: } else {
// CHECK: %[[OUTER_FAIL:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[OUTER_FAIL]]#0, %[[OUTER_FAIL]]#1
// CHECK: tvm_ffi.return %[[OUTER]]#0, %[[OUTER]]#1
tvm_ffi.func @nested_mixed(%input: !tvm_ffi.monad.result<i64>, %unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %outer:2 = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  ^bb0(%payload: i64):
    %inner:2 = tvm_ffi.monad.bind %unit : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  ^bb_unit(%unit_payload: !tvm_ffi.none):
      %result_none = tvm_ffi.constant.none
      %result:2 = tvm_ffi.monad.pure %payload, %result_none : (i64, !tvm_ffi.none) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>)
      tvm_ffi.monad.yield %result#0, %result#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
    }
    tvm_ffi.monad.yield %inner#0, %inner#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
  }

  tvm_ffi.return %outer#0, %outer#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
}
