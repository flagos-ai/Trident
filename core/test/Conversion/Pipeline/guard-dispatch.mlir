// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi_'  --implicit-check-not='unrealized_conversion_cast'

// Guards select an execution entry with the same original argument layout.
// A shared prefix guards two ordered leaves. The original pack is forwarded;
// nested input adaptation belongs to execution.
// CHECK-LABEL: tvm_ffi_abi.pack.func @dispatch(
// CHECK-SAME: %[[ARGS:[a-zA-Z0-9_]+]]:
// CHECK: scf.execute_region
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get
// CHECK: %[[ROOT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[INPUT]]
// CHECK: tvm_ffi_abi.call @check(%[[ROOT]])
// CHECK: tvm_ffi_abi.call @check_branch(%[[ROOT]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.call @run(%[[ARGS]])
// CHECK: tvm_ffi_abi.object.inc_ref %[[VALUE]]
// CHECK: tvm_ffi_abi.object.dec_ref %[[VALUE]]
// CHECK: scf.yield %[[OK]], %[[VALUE]]
// CHECK: tvm_ffi_abi.pack.call @run_fallback(%[[ARGS]])
// LLVM-LABEL: llvm.func @__tvm_ffi_dispatch(
// LLVM: llvm.call @__tvm_ffi_check
// LLVM: llvm.cond_br
// LLVM: llvm.call @__tvm_ffi_run
// LLVM: llvm.call @__tvm_ffi_run_fallback
// LLVM: llvm.call @TVMFFIObjectDecRef
// LLVM: llvm.return

tvm_ffi_api.pack.func @dispatch(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_api.any {
  %success, %payload = scf.execute_region -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
    %index = arith.constant 0 : i64
    %input = tvm_ffi_api.pack.get %args[%index] : (!tvm_ffi_abi.pack.arguments, i64) -> !tvm_ffi_api.any
    %inputs = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>
    %result = tvm_ffi_monad.call @check(%inputs) : (!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
    %ready, %value = tvm_ffi_monad.unpack %result : (!tvm_ffi_monad.result<!tvm_ffi_abi.bool>) -> (i1, !tvm_ffi_api.any)
    cf.cond_br %ready, ^predicate, ^failed
  ^predicate:
    %matched = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_abi.bool
    %test = tvm_ffi_api.get %matched : !tvm_ffi_abi.bool -> i1
    cf.cond_br %test, ^branch, ^miss
  ^branch:
    %branch_result = tvm_ffi_monad.call @check_branch(%inputs) : (!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
    %branch_ready, %branch_value = tvm_ffi_monad.unpack %branch_result : (!tvm_ffi_monad.result<!tvm_ffi_abi.bool>) -> (i1, !tvm_ffi_api.any)
    cf.cond_br %branch_ready, ^branch_predicate, ^failed
  ^branch_predicate:
    %branch_matched = tvm_ffi_api.downcast %branch_value : !tvm_ffi_api.any -> !tvm_ffi_abi.bool
    %branch_test = tvm_ffi_api.get %branch_matched : !tvm_ffi_abi.bool -> i1
    cf.cond_br %branch_test, ^execute, ^execute_fallback
  ^failed:
    %failed_ok = tvm_ffi_abi.constant.bool false
    %none = tvm_ffi_abi.constant.none
    %empty = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
    scf.yield %failed_ok, %empty : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  ^execute:
    %ok, %answer = tvm_ffi_api.pack.call @run(%args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
    scf.yield %ok, %answer : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  ^execute_fallback:
    %fallback_ok, %fallback_answer = tvm_ffi_api.pack.call @run_fallback(%args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
    scf.yield %fallback_ok, %fallback_answer : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  ^miss:
    %yes = tvm_ffi_abi.constant.bool true
    %error = tvm_ffi_api.exception "GuardMiss" : !tvm_ffi_api.exception
    %sentinel = tvm_ffi_api.upcast %error : !tvm_ffi_api.exception -> !tvm_ffi_api.any
    scf.yield %yes, %sentinel : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  } {no_inline}
  tvm_ffi_api.return %success, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

tvm_ffi_api.func private @check(!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_abi.bool
tvm_ffi_api.func private @check_branch(!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_abi.bool
tvm_ffi_api.func private @run(!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_api.any
tvm_ffi_api.func private @run_fallback(!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>>) -> !tvm_ffi_api.any
