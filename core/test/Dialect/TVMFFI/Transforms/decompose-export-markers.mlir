// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each > %t
// RUN: trident-core-opt %t --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// CHECK-LABEL:   func.func @container_branch(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.c.array) {
// CHECK:             scf.yield {tvm_ffi.retain} %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             scf.yield {tvm_ffi.retain} %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           return {tvm_ffi.retain} %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @continuation(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]], %[[TO_0]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[ARG0]], %[[TO_1]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[CONSTANT_3]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:             scf.yield {tvm_ffi.retain} %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]](%[[CONSTANT_5]], %[[UPCAST_1]]) {tvm_ffi.cleanup} : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_6]] : !tvm_ffi.object
// CHECK:             scf.yield {tvm_ffi.retain} %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           return {tvm_ffi.retain} %[[IF_0]] : !tvm_ffi.c.array






// Converted function returns and SCF yields preserve pending markers.
func.func @container_branch(%predicate: i1, %input: !tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  }
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<!tvm_ffi.tensor>
}

// Bind success yields preserve intent; generated failure yields remain unmarked.
func.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%payload: !tvm_ffi.tensor):
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  }
  return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
