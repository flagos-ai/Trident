// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: llvm.func @invoke(
// CHECK: llvm.call @__tvm_ffi_identity
// CHECK: llvm.return
func.func @invoke(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  %result = tvm_ffi.monad.call @identity(%input) {arg_attrs = [{test.argument}], res_attrs = [{test.result}]} : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
}

// CHECK-LABEL: llvm.func @invoke_function(
// CHECK: llvm.ptrtoint
// CHECK: llvm.call @__tvm_ffi_function_identity
// CHECK: llvm.return
func.func @invoke_function(%input: !tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function> {
  %result = tvm_ffi.monad.call @function_identity(%input) : (!tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_identity(
// CHECK-NOT: llvm.call @identity
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @__tvm_ffi_function_identity(
// CHECK: llvm.inttoptr
// CHECK: llvm.ptrtoint
// CHECK-NOT: llvm.call @function_identity
// CHECK: llvm.return
tvm_ffi.func @identity(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi} {
  %any = tvm_ffi.upcast %input : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}

// Function conversion fills the compatible declaration without duplicating it.
func.func private @__tvm_ffi_identity(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

tvm_ffi.func @function_identity(%input: !tvm_ffi.c.function) -> !tvm_ffi.c.function attributes {emit_tvm_ffi_abi} {
  %any = tvm_ffi.upcast %input : !tvm_ffi.c.function -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}

// A dispatcher forwards a borrowed argument pack to typed packed functions. Failed ABI
// calls propagate; only successful Exception payloads trigger the next attempt.
// CHECK-LABEL: llvm.func @__tvm_ffi_dispatch(
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @__tvm_ffi_identity(%[[CTX:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[COUNT:[a-zA-Z0-9_]+]], %[[OUT:[a-zA-Z0-9_]+]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS]], %[[ZERO:[a-zA-Z0-9_]+]]
// CHECK: %[[BITS:[a-zA-Z0-9_]+]] = llvm.zext %[[OK]]
// CHECK: llvm.cond_br %[[OK]], ^[[LOAD:bb[0-9]+]], ^[[JOIN:bb[0-9]+]](%[[NONE:[a-zA-Z0-9_]+]] :
// CHECK: ^[[LOAD]]:
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[OUT]]
// CHECK-NEXT: llvm.br ^[[JOIN]](%[[VALUE]] :
// CHECK: ^[[JOIN]](%[[PAYLOAD:[a-zA-Z0-9_]+]]:
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = llvm.trunc %[[BITS]]
// CHECK-NEXT: llvm.cond_br %[[READY]], ^[[TEST:bb[0-9]+]], ^[[FAILED:bb[0-9]+]]
// CHECK: ^[[TEST]]:
// CHECK-NEXT: %[[TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[PAYLOAD]][0]
// CHECK-NEXT: %[[REJECTED:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[TAG]], %[[INDEX:[a-zA-Z0-9_]+]]
// CHECK-NEXT: llvm.cond_br %[[REJECTED]], ^[[RETRY:bb[0-9]+]], ^[[DONE:bb[0-9]+]]
// CHECK: ^[[RETRY]]:
// CHECK: llvm.call @__tvm_ffi_identity
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: ^[[FAILED]]:
// CHECK-NOT: llvm.call @__tvm_ffi_identity
// CHECK: llvm.return
tvm_ffi.pack.func @dispatch(%input: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
  %ok, %value = tvm_ffi.pack.call @identity(%input) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
  %ready = tvm_ffi.get %ok : !tvm_ffi.bool -> i1
  %success, %payload = scf.if %ready -> (!tvm_ffi.bool, !tvm_ffi.any) {
    %rejected = tvm_ffi.isa %value is !tvm_ffi.exception : !tvm_ffi.any
    %retry_ok, %retry_value = scf.if %rejected -> (!tvm_ffi.bool, !tvm_ffi.any) {
      %next_ok, %next_value = tvm_ffi.pack.call @identity(%input) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %next_ok, %next_value : !tvm_ffi.bool, !tvm_ffi.any
    } else {
      scf.yield %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
    }
    scf.yield %retry_ok, %retry_value : !tvm_ffi.bool, !tvm_ffi.any
  } else {
    scf.yield %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
  }
  tvm_ffi.return %success, %payload : !tvm_ffi.bool, !tvm_ffi.any
}
