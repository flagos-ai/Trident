// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// Guards select an execution entry with the same original argument layout.
// A shared prefix guards two ordered leaves. The original pack is forwarded;
// nested input adaptation belongs to execution.
// CHECK-LABEL: tvm_ffi.pack.func @dispatch(
// CHECK-SAME: %[[ARGS:[a-zA-Z0-9_]+]]:
// CHECK: scf.execute_region
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get
// CHECK: %[[ROOT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]]
// CHECK: tvm_ffi.call @check(%[[ROOT]])
// CHECK: tvm_ffi.call @check_branch(%[[ROOT]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.call @run(%[[ARGS]])
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]]
// CHECK: tvm_ffi.ObjectDecRef %[[VALUE]]
// CHECK: scf.yield %[[OK]], %[[VALUE]]
// CHECK: tvm_ffi.pack.call @run_fallback(%[[ARGS]])
// LLVM-LABEL: llvm.func @__tvm_ffi_dispatch(
// LLVM: llvm.call @__tvm_ffi_check
// LLVM: llvm.cond_br
// LLVM: llvm.call @__tvm_ffi_run
// LLVM: llvm.call @__tvm_ffi_run_fallback
// LLVM: llvm.call @TVMFFIObjectDecRef
// LLVM: llvm.return

tvm_ffi.pack.func @dispatch(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
  %success, %payload = scf.execute_region -> (!tvm_ffi.bool, !tvm_ffi.any) {
    %index = arith.constant 0 : i64
    %input = tvm_ffi.pack.get %args[%index] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.any
    %inputs = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>
    %result = tvm_ffi.monad.call @check(%inputs) : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
    %ready, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.bool>) -> (i1, !tvm_ffi.any)
    cf.cond_br %ready, ^predicate, ^failed
  ^predicate:
    %matched = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.bool
    %test = tvm_ffi.get %matched : !tvm_ffi.bool -> i1
    cf.cond_br %test, ^branch, ^miss
  ^branch:
    %branch_result = tvm_ffi.monad.call @check_branch(%inputs) : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
    %branch_ready, %branch_value = tvm_ffi.monad.unpack %branch_result : (!tvm_ffi.monad.result<!tvm_ffi.bool>) -> (i1, !tvm_ffi.any)
    cf.cond_br %branch_ready, ^branch_predicate, ^failed
  ^branch_predicate:
    %branch_matched = tvm_ffi.downcast %branch_value : !tvm_ffi.any -> !tvm_ffi.bool
    %branch_test = tvm_ffi.get %branch_matched : !tvm_ffi.bool -> i1
    cf.cond_br %branch_test, ^execute, ^execute_fallback
  ^failed:
    %failed_ok = tvm_ffi.constant.bool false
    %none = tvm_ffi.constant.none
    %empty = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
    scf.yield %failed_ok, %empty : !tvm_ffi.bool, !tvm_ffi.any
  ^execute:
    %ok, %answer = tvm_ffi.pack.call @run(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
    scf.yield %ok, %answer : !tvm_ffi.bool, !tvm_ffi.any
  ^execute_fallback:
    %fallback_ok, %fallback_answer = tvm_ffi.pack.call @run_fallback(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
    scf.yield %fallback_ok, %fallback_answer : !tvm_ffi.bool, !tvm_ffi.any
  ^miss:
    %yes = tvm_ffi.constant.bool true
    %error = tvm_ffi.exception "GuardMiss" : !tvm_ffi.exception
    %sentinel = tvm_ffi.upcast %error : !tvm_ffi.exception -> !tvm_ffi.any
    scf.yield %yes, %sentinel : !tvm_ffi.bool, !tvm_ffi.any
  } {no_inline}
  tvm_ffi.return %success, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

tvm_ffi.func private @check(!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.bool
tvm_ffi.func private @check_branch(!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.bool
tvm_ffi.func private @run(!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.any
tvm_ffi.func private @run_fallback(!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.any
