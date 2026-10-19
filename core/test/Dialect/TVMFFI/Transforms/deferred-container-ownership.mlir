// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// Borrowed container arguments are retained on export, never scheduled for release.
// CHECK-LABEL: func.func @borrowed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: return %[[INPUT]], %[[INPUT]]
// LLVM-LABEL: llvm.func @borrowed(
func.func @borrowed(%input: !tvm_ffi.tuple<>) -> (!tvm_ffi.tuple<>, !tvm_ffi.tuple<>) {
  return {tvm_ffi.retain} %input, %input : !tvm_ffi.tuple<>, !tvm_ffi.tuple<>
}

// Every payload occurrence receives a separate reference after CArray erasure.
// Both Result references and the independently constructed container are released.
// CHECK-LABEL: func.func @repeated_payloads(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]()
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ARRAY]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ARRAY]]
// CHECK-NEXT: %[[RESULTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[ARRAY]], %[[ARRAY]]
// CHECK-NEXT: %[[SECOND_OK:[a-zA-Z0-9_]+]], %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULTS]]#1
// CHECK-NEXT: scf.if %[[SECOND_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.any
// CHECK-NEXT: }
// CHECK: %[[FIRST_OK:[a-zA-Z0-9_]+]], %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULTS]]#0
// CHECK-NEXT: scf.if %[[FIRST_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.any
// CHECK-NEXT: }
// CHECK: tvm_ffi.ObjectDecRef %[[ARRAY]] : !tvm_ffi.c.array
// CHECK-NEXT: return
// LLVM-LABEL: llvm.func @repeated_payloads(
func.func @repeated_payloads() {
  %array = tvm_ffi.array.create() : () -> !tvm_ffi.array<!tvm_ffi.int>
  %results:2 = tvm_ffi.monad.pure %array, %array {tvm_ffi.cleanup, tvm_ffi.retain_payloads} : (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>, !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>)
  return {tvm_ffi.retain}
}

// Extraction creates an independent reference even without a frontend marker.
// A cast of that reference borrows it and does not acquire cleanup responsibility.
// CHECK-LABEL: func.func @nested(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[INPUT]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CHECK-NEXT: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[ELEMENT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ALIAS]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CHECK-NEXT: return %[[ALIAS]]
// LLVM-LABEL: llvm.func @nested(
func.func @nested(%input: !tvm_ffi.array<!tvm_ffi.tuple<>>) -> !tvm_ffi.any {
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %input[%index] : !tvm_ffi.array<!tvm_ffi.tuple<>> -> !tvm_ffi.tuple<>
  %alias = tvm_ffi.upcast %element : !tvm_ffi.tuple<> -> !tvm_ffi.any
  return {tvm_ffi.retain} %alias : !tvm_ffi.any
}

// Result exports retain widened payloads only inside a success branch.
// CHECK-LABEL: func.func @result_export(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK-NEXT: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[INPUT]]
// LLVM-LABEL: llvm.func @result_export(
func.func @result_export(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<>> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.tuple<>>
}

// Conditional yields establish the receiving scope's reference before its cleanup.
// CHECK-LABEL: func.func @conditional(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK-NEXT: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: scf.yield %[[INPUT]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: scf.yield %[[INPUT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OUTPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTPUT]]
// CHECK-NEXT: return %[[OUTPUT]]
// LLVM-LABEL: llvm.func @conditional(
func.func @conditional(%predicate: i1, %input: !tvm_ffi.tuple<>) -> !tvm_ffi.tuple<> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<>
}
