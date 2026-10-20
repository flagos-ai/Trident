// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// CHECK-LABEL:   func.func @borrowed(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> (!tvm_ffi.c.array, !tvm_ffi.c.array) {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           return %[[ARG0]], %[[ARG0]] : !tvm_ffi.c.array, !tvm_ffi.c.array
// CHECK-LABEL:   func.func @repeated_payloads() {
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]]() : () -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_1]] : !tvm_ffi.c.array -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_1]] : !tvm_ffi.c.array -> !tvm_ffi.any
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[CONSTANT_1]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @nested(
// CHECK-SAME:                      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.any {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_0]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[VAL_1]] : !tvm_ffi.c.array -> !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectIncRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return %[[UPCAST_0]] : !tvm_ffi.any
// CHECK-LABEL:   func.func @result_export(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           return %[[ARG0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @conditional(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.c.array) {
// CHECK:             tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array






// Borrowed container arguments are retained on export, never scheduled for release.
// LLVM-LABEL: llvm.func @borrowed(
func.func @borrowed(%input: !tvm_ffi.tuple<>) -> (!tvm_ffi.tuple<>, !tvm_ffi.tuple<>) {
  return {tvm_ffi.retain} %input, %input : !tvm_ffi.tuple<>, !tvm_ffi.tuple<>
}

// Every payload occurrence receives a separate reference after CArray erasure.
// Both Result references and the independently constructed container are released.
// LLVM-LABEL: llvm.func @repeated_payloads(
func.func @repeated_payloads() {
  %array = tvm_ffi.array.create() {tvm_ffi.cleanup} : () -> !tvm_ffi.array<!tvm_ffi.int>
  %results:2 = tvm_ffi.monad.pure %array, %array {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>, !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>)
  return {tvm_ffi.retain}
}

// Extraction creates an independent reference with explicit cleanup intent.
// A cast of that reference borrows it and does not acquire cleanup responsibility.
// LLVM-LABEL: llvm.func @nested(
func.func @nested(%input: !tvm_ffi.array<!tvm_ffi.tuple<>>) -> !tvm_ffi.any {
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %input[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tuple<>> -> !tvm_ffi.tuple<>
  %alias = tvm_ffi.upcast %element : !tvm_ffi.tuple<> -> !tvm_ffi.any
  return {tvm_ffi.retain} %alias : !tvm_ffi.any
}

// Result exports retain widened payloads only inside a success branch.
// LLVM-LABEL: llvm.func @result_export(
func.func @result_export(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<>> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.tuple<>>
}

// Conditional yields establish the receiving scope's reference before its cleanup.
// LLVM-LABEL: llvm.func @conditional(
func.func @conditional(%predicate: i1, %input: !tvm_ffi.tuple<>) -> !tvm_ffi.tuple<> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<>
}
