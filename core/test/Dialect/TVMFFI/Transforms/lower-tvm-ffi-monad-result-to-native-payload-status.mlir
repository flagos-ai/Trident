// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @arguments(
// CHECK-SAME: %[[PAYLOAD:[a-zA-Z0-9_]+]]: i64, %[[OK:[a-zA-Z0-9_]+]]: i1, %[[NONE_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.none, %[[UNIT:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<i1>)
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]]:2 = scf.if %[[OK]] -> (i64, i1)
// CHECK: scf.yield %[[PAYLOAD]],
// CHECK: %[[UNIT_MATCH:[a-zA-Z0-9_]+]] = scf.if %[[UNIT]] -> (i1)
// CHECK: %[[RETURN_0:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#0, %[[TRUE:[a-zA-Z0-9_]+]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[RETURN_1:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#1, %[[TRUE:[a-zA-Z0-9_]+]] : i1, i1 to !tvm_ffi.monad.result<i1>
// CHECK: %[[RETURN_2:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[UNIT_MATCH]], %[[TRUE:[a-zA-Z0-9_]+]] : i1, i1 to !tvm_ffi.monad.result<i1>
// CHECK: tvm_ffi.return %[[RETURN_0]], %[[RETURN_1]], %[[RETURN_2]]
tvm_ffi.func @arguments(%input: !tvm_ffi.monad.result<i64>, %unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<i1>) {
  %payload_value_ok, %payload_value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  %value, %ok = scf.if %payload_value_ok -> (i64, i1) {
    %yes = arith.constant true
    scf.yield %payload_value, %yes : i64, i1
  } else {
    %zero = arith.constant 0 : i64
    %no = arith.constant false
    scf.yield %zero, %no : i64, i1
  }
  %unit_payload_ok, %unit_payload = tvm_ffi.monad.unpack %unit : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  %unit_ok = scf.if %unit_payload_ok -> (i1) {
    %yes = arith.constant true
    scf.yield %yes : i1
  } else {
    %no = arith.constant false
    scf.yield %no : i1
  }

  %returned:3 = tvm_ffi.monad.pure %value, %ok, %unit_ok : (i64, i1, i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<i1>)
  tvm_ffi.return %returned#0, %returned#1, %returned#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>, !tvm_ffi.monad.result<i1>
}

// CHECK-LABEL: tvm_ffi.func @block_argument(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[STATUS:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>)
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[VALUE]], %[[STATUS]] : i64, i1)
// CHECK: ^[[NEXT]](%[[PAYLOAD:[a-zA-Z0-9_]+]]: i64, %[[OK:[a-zA-Z0-9_]+]]: i1):
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]]:2 = scf.if %[[OK]] -> (i64, i1)
// CHECK: scf.yield %[[PAYLOAD]],
// CHECK: %[[RETURN_0:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#0, %[[TRUE:[a-zA-Z0-9_]+]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[RETURN_1:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#1, %[[TRUE:[a-zA-Z0-9_]+]] : i1, i1 to !tvm_ffi.monad.result<i1>
// CHECK: tvm_ffi.return %[[RETURN_0]], %[[RETURN_1]]
tvm_ffi.func @block_argument(%input: !tvm_ffi.monad.result<i64>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>) {
  cf.br ^next(%input : !tvm_ffi.monad.result<i64>)
^next(%forwarded: !tvm_ffi.monad.result<i64>):
  %payload_value_ok, %payload_value = tvm_ffi.monad.unpack %forwarded : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  %value, %ok = scf.if %payload_value_ok -> (i64, i1) {
    %yes = arith.constant true
    scf.yield %payload_value, %yes : i64, i1
  } else {
    %zero = arith.constant 0 : i64
    %no = arith.constant false
    scf.yield %zero, %no : i64, i1
  }

  %returned:2 = tvm_ffi.monad.pure %value, %ok : (i64, i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>)
  tvm_ffi.return %returned#0, %returned#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>
}

// CHECK-LABEL: tvm_ffi.func @index_switch(
// CHECK-SAME: %[[SELECTOR:[a-zA-Z0-9_]+]]: index, %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>)
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY]] -> (i64, i1)
// CHECK: %[[SELECTED:[a-zA-Z0-9_]+]]:2 = scf.index_switch %[[SELECTOR]] -> i64, i1
// CHECK: case 0 {
// CHECK: scf.yield %[[INPUT]]#0, %[[INPUT]]#1 : i64, i1
// CHECK: default {
// CHECK: scf.yield %[[INPUT]]#0, %[[INPUT]]#1 : i64, i1
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]]:2 = scf.if %[[SELECTED]]#1 -> (i64, i1)
// CHECK: scf.yield %[[SELECTED]]#0,
// CHECK: %[[RETURN_0:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#0, %[[TRUE:[a-zA-Z0-9_]+]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[RETURN_1:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[MATCH]]#1, %[[TRUE:[a-zA-Z0-9_]+]] : i1, i1 to !tvm_ffi.monad.result<i1>
// CHECK: tvm_ffi.return %[[RETURN_0]], %[[RETURN_1]]
tvm_ffi.func @index_switch(%selector: index, %value: i64, %ready: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>) {
  %input = scf.if %ready -> (!tvm_ffi.monad.result<i64>) {
    %input_pure = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    scf.yield %input_pure : !tvm_ffi.monad.result<i64>
  } else {
    %input_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
    scf.yield %input_fail : !tvm_ffi.monad.result<i64>
  }
  %selected = scf.index_switch %selector -> !tvm_ffi.monad.result<i64>
    case 0 {
      scf.yield %input : !tvm_ffi.monad.result<i64>
    }
    default {
      scf.yield %input : !tvm_ffi.monad.result<i64>
    }
  %payload_value_ok, %payload_value = tvm_ffi.monad.unpack %selected : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  %payload, %ok = scf.if %payload_value_ok -> (i64, i1) {
    %yes = arith.constant true
    scf.yield %payload_value, %yes : i64, i1
  } else {
    %zero = arith.constant 0 : i64
    %no = arith.constant false
    scf.yield %zero, %no : i64, i1
  }

  %returned:2 = tvm_ffi.monad.pure %payload, %ok : (i64, i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>)
  tvm_ffi.return %returned#0, %returned#1 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i1>
}

// CHECK-LABEL: tvm_ffi.func @mixed(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: i64, %[[SECOND:[a-zA-Z0-9_]+]]: i32, %[[READY:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>)
// CHECK: %[[RESULTS:[a-zA-Z0-9_]+]]:6 = scf.if %[[READY]] -> (i64, i1, !tvm_ffi.none, i1, i32, i1) {
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: scf.yield %[[FIRST]], %[[TRUE]], %[[NONE]], %[[TRUE]], %[[SECOND]], %[[TRUE]] : i64, i1, !tvm_ffi.none, i1, i32, i1
// CHECK: } else {
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK: %[[FIRST_POISON:[a-zA-Z0-9_]+]] = ub.poison : i64
// CHECK: %[[NONE_ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[NONE_FAILED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE_ZERO]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.none
// CHECK: %[[SECOND_POISON:[a-zA-Z0-9_]+]] = ub.poison : i32
// CHECK: scf.yield %[[FIRST_POISON]], %[[FALSE]], %[[NONE_FAILED]], %[[FALSE]], %[[SECOND_POISON]], %[[FALSE]] : i64, i1, !tvm_ffi.none, i1, i32, i1
// CHECK: %[[FIRST_RESULT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULTS]]#0, %[[RESULTS]]#1 : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[UNIT_RESULT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULTS]]#2, %[[RESULTS]]#3 : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: %[[SECOND_RESULT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULTS]]#4, %[[RESULTS]]#5 : i32, i1 to !tvm_ffi.monad.result<i32>
// CHECK: tvm_ffi.return %[[FIRST_RESULT]], %[[UNIT_RESULT]], %[[SECOND_RESULT]]
tvm_ffi.func @mixed(%first: i64, %second: i32, %ready: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>) {
  %results:3 = scf.if %ready -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>) {
    %results_pure_none = tvm_ffi.constant.none
    %results_pure:3 = tvm_ffi.monad.pure %first, %results_pure_none, %second : (i64, !tvm_ffi.none, i32) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>)
    scf.yield %results_pure#0, %results_pure#1, %results_pure#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
  } else {
    %results_fail:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
    scf.yield %results_fail#0, %results_fail#1, %results_fail#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
  }

  tvm_ffi.return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i32>
}
