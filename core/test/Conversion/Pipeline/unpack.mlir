// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not="!tvm_ffi."

// CHECK-LABEL:   tvm_ffi.func @unpack(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> !tvm_ffi.any {
// CHECK:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[ARG0]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[TUPLE_0]]#1) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           tvm_ffi.ObjectDecRef %[[TUPLE_0]]#1 : !tvm_ffi.any
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[TUPLE_1]] {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_1]] : i1 -> !tvm_ffi.bool
// CHECK:           tvm_ffi.return {tvm_ffi.retain} %[[TO_0]], %[[TUPLE_2]]#1 : !tvm_ffi.bool, !tvm_ffi.any


tvm_ffi.func @unpack(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.any {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  %return_ok_1, %return_value_1 = tvm_ffi.monad.unpack %returned : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  %return_flag_1 = tvm_ffi.to %return_ok_1 : i1 -> !tvm_ffi.bool
  tvm_ffi.return {tvm_ffi.retain} %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}

// LLVM-LABEL: llvm.func @unpack(
// LLVM: llvm.return
