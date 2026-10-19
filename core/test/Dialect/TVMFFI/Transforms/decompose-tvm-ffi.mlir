// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not=tvm_ffi.monad.bind --implicit-check-not=tvm_ffi.monad.yield

// CHECK-LABEL: tvm_ffi.func @aggregate(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64,
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]] : !tvm_ffi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[NATIVE]], %[[VALUE]]
// CHECK: %[[SUM_BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[SUM]] : i64 -> !tvm_ffi.int
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[SUM_BOXED]]
// CHECK: scf.yield %[[RESULT]]
// CHECK: } else {
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[FAILED]]
// CHECK: tvm_ffi.return %[[OUTPUT]]
tvm_ffi.func @aggregate(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
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

  tvm_ffi.return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// CHECK-LABEL: tvm_ffi.func @external
tvm_ffi.func @external(!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>

// Decomposition is independent of function ABI lowering and preserves captures.
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: %[[TENSOR_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.tensor>, %[[SCALAR_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[CAPTURE:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[TENSOR_OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[TENSOR_INPUT]]
// CHECK: %[[SCALAR_OK:[a-zA-Z0-9_]+]], %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[SCALAR_INPUT]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_OK]], %[[SCALAR_OK]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]]
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK: %[[SCALAR_BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[SCALAR]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[SCALAR_NATIVE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR_BOXED]] : !tvm_ffi.int -> i64
// CHECK: %[[SUM:[a-zA-Z0-9_]+]] = arith.addi %[[SCALAR_NATIVE]], %[[CAPTURE]]
// CHECK: %[[SUM_BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[SUM]] : i64 -> !tvm_ffi.int
// CHECK: %[[PURE:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[TENSOR]], %[[SUM_BOXED]]
// CHECK: scf.yield %[[PURE]]#0, %[[PURE]]#1
// CHECK: } else {
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[FAILED]]#0, %[[FAILED]]#1
// CHECK: return %[[OUTPUT]]#0, %[[OUTPUT]]#1
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

// CHECK-LABEL: func.func @nested_mixed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[UNIT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if %[[OK]]
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]] : !tvm_ffi.int -> i64
// CHECK: %[[UNIT_OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[UNIT]]
// CHECK: %[[INNER:[a-zA-Z0-9_]+]]:2 = scf.if %[[UNIT_OK]]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: %[[UNIT_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[REBOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[NATIVE]] : i64 -> !tvm_ffi.int
// CHECK: %[[PURE:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[REBOXED]], %[[UNIT_VALUE]]
// CHECK: scf.yield %[[PURE]]#0, %[[PURE]]#1
// CHECK: } else {
// CHECK: %[[INNER_FAIL:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[INNER_FAIL]]#0, %[[INNER_FAIL]]#1
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1
// CHECK: } else {
// CHECK: %[[OUTER_FAIL:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.fail
// CHECK: scf.yield %[[OUTER_FAIL]]#0, %[[OUTER_FAIL]]#1
// CHECK: return %[[OUTER]]#0, %[[OUTER]]#1
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
