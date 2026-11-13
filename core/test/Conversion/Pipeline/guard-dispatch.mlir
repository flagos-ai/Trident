// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// Python emits this shape directly: fallible checks and graph input adaptation live
// in a guard-selecting execute_region, with payload exits dominated by their producers.
// CHECK-LABEL: tvm_ffi.pack.func @dispatch(
// CHECK: scf.execute_region
// CHECK: tvm_ffi.pack.size
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get
// CHECK: %[[ROOT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]]
// CHECK: tvm_ffi.call @check(%[[ROOT]])
// CHECK: %[[ZERO1:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[INDEX1:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO1]] : i64 -> !tvm_ffi.int
// CHECK: %[[GET1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[NESTED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET1]] : !tvm_ffi.c.function(%[[ROOT]], %[[INDEX1]])
// CHECK: %[[ZERO2:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[INDEX2:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO2]] : i64 -> !tvm_ffi.int
// CHECK: %[[GET2:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET2]] : !tvm_ffi.c.function(%[[NESTED]], %[[INDEX2]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @run(%[[TENSOR]])
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]]
// CHECK: tvm_ffi.ObjectDecRef %[[VALUE]]
// CHECK: scf.yield %[[OK]], %[[VALUE]]
// LLVM-LABEL: llvm.func @__tvm_ffi_dispatch(
// LLVM: llvm.call @__tvm_ffi_check
// LLVM: llvm.cond_br
// LLVM: llvm.call @__tvm_ffi_run
// LLVM: llvm.call @TVMFFIObjectDecRef
// LLVM: llvm.return

tvm_ffi.pack.func @dispatch(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
  %success, %payload = scf.execute_region -> (!tvm_ffi.bool, !tvm_ffi.any) {
    %count = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
    %one = arith.constant 1 : i64
    %matches = arith.cmpi eq, %count, %one : i64
    cf.cond_br %matches, ^slot, ^miss
  ^slot:
    %index = arith.constant 0 : i64
    %input = tvm_ffi.pack.get %args[%index] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.any
    %tag = tvm_ffi.isa %input is !tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>> : !tvm_ffi.any
    cf.cond_br %tag, ^check, ^miss
  ^check:
    %inputs = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>
    %result = tvm_ffi.monad.call @check(%inputs) : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
    %ready, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.bool>) -> (i1, !tvm_ffi.any)
    cf.cond_br %ready, ^predicate, ^failed
  ^predicate:
    %matched = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.bool
    %test = tvm_ffi.get %matched : !tvm_ffi.bool -> i1
    cf.cond_br %test, ^execute, ^miss
  ^failed:
    %failed_ok = tvm_ffi.to %ready : i1 -> !tvm_ffi.bool
    %none = tvm_ffi.constant.none
    %empty = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
    scf.yield %failed_ok, %empty : !tvm_ffi.bool, !tvm_ffi.any
  ^execute:
    %nested = tvm_ffi.tuple.unconstruct %inputs : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.tuple<!tvm_ffi.tensor>
    %tensor = tvm_ffi.tuple.unconstruct %nested : (!tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tensor
    %graph = tvm_ffi.monad.call @run(%tensor) : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.any>
    %called, %answer = tvm_ffi.monad.unpack %graph : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
    %ok = tvm_ffi.to %called : i1 -> !tvm_ffi.bool
    scf.yield %ok, %answer : !tvm_ffi.bool, !tvm_ffi.any
  ^miss:
    %yes = tvm_ffi.constant.bool true
    %error = tvm_ffi.exception "GuardMiss" : !tvm_ffi.exception
    %sentinel = tvm_ffi.upcast %error : !tvm_ffi.exception -> !tvm_ffi.any
    scf.yield %yes, %sentinel : !tvm_ffi.bool, !tvm_ffi.any
  } {no_inline}
  tvm_ffi.return %success, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

tvm_ffi.func private @check(!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.bool
tvm_ffi.func private @run(!tvm_ffi.tensor) -> !tvm_ffi.any
