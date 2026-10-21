// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not="!tvm_ffi."

// CHECK-LABEL:   tvm_ffi.func @short_circuit(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !llvm.ptr) -> !tvm_ffi.int {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant 7 : i64
// CHECK:             %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_0]] : i64 -> !tvm_ffi.int
// CHECK-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[TO_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_2]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.int
// CHECK:             %[[LOAD_0:[a-zA-Z0-9_$.-]+]] = llvm.load %[[ARG1]] : !llvm.ptr -> i64
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:             %[[ADDI_0:[a-zA-Z0-9_$.-]+]] = arith.addi %[[LOAD_0]], %[[CONSTANT_5]] : i64
// CHECK:             llvm.store %[[ADDI_0]], %[[ARG1]] : i64, !llvm.ptr
// CHECK-DAG:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.int -> !tvm_ffi.any
// CHECK-DAG:             %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_7]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_8]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_5:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_5]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_1]] : i1 -> !tvm_ffi.bool
// CHECK:           tvm_ffi.return %[[TO_1]], %[[TUPLE_5]]#1 : !tvm_ffi.bool, !tvm_ffi.any


// A failed Result skips a continuation with an independent side effect.
tvm_ffi.func @short_circuit(%ready: i1, %counter: !llvm.ptr) -> !tvm_ffi.int {
  %input = scf.if %ready -> !tvm_ffi.monad.result<!tvm_ffi.int> {
    %seven = arith.constant 7 : i64
    %seven_boxed = tvm_ffi.to %seven : i64 -> !tvm_ffi.int
    %pure = tvm_ffi.monad.pure %seven_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.int>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%value_boxed: !tvm_ffi.int):
    %old = llvm.load %counter : !llvm.ptr -> i64
    %one = arith.constant 1 : i64
    %new = arith.addi %old, %one : i64
    llvm.store %new, %counter : i64, !llvm.ptr
    %result = tvm_ffi.monad.pure %value_boxed : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  %return_ok_1, %return_value_1 = tvm_ffi.monad.unpack %output : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.any)
  %return_flag_1 = tvm_ffi.to %return_ok_1 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}

// LLVM-LABEL: llvm.func @short_circuit(
// LLVM: llvm.return
