// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi_'  --implicit-check-not='unrealized_conversion_cast'
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --symbol-dce --lower-tvm-ffi-abi-to-pack --convert-tvm-ffi-abi-to-func --canonicalize --inline --verify-each | FileCheck %s --check-prefix=STRUCTURED

// STRUCTURED-LABEL: func.func @invoke(
// STRUCTURED-NOT: call @__tvm_ffi_identity
// STRUCTURED: scf.if
// STRUCTURED-LABEL: func.func @invoke_function(
// STRUCTURED-NOT: call @__tvm_ffi_function_identity
// STRUCTURED: scf.if
// STRUCTURED-LABEL: func.func @__tvm_ffi_dispatch(
// STRUCTURED: scf.if
// STRUCTURED: call @__tvm_ffi_identity
// STRUCTURED: scf.if


// CHECK-LABEL: llvm.func @invoke(
// CHECK-NOT: llvm.call @__tvm_ffi_identity
// CHECK: llvm.return
func.func @invoke(%input: !tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor> {
  %result = tvm_ffi_monad.call @identity(%input) {arg_attrs = [{test.argument}], res_attrs = [{test.result}]} : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
}

// CHECK-LABEL: llvm.func @invoke_function(
// CHECK: llvm.ptrtoint
// CHECK-NOT: llvm.call @__tvm_ffi_function_identity
// CHECK: llvm.return
func.func @invoke_function(%input: !tvm_ffi_abi.function) -> !tvm_ffi_monad.result<!tvm_ffi_abi.function> {
  %result = tvm_ffi_monad.call @function_identity(%input) : (!tvm_ffi_abi.function) -> !tvm_ffi_monad.result<!tvm_ffi_abi.function>
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.function>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_identity(
// CHECK-NOT: llvm.call @identity
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @__tvm_ffi_function_identity(
// CHECK: llvm.inttoptr
// CHECK: llvm.ptrtoint
// CHECK-NOT: llvm.call @function_identity
// CHECK: llvm.return
tvm_ffi_api.func @identity(%input: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor {
  %any = tvm_ffi_api.upcast %input : !tvm_ffi_abi.tensor -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// Function conversion fills the compatible declaration without duplicating it.
func.func private @__tvm_ffi_identity(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

tvm_ffi_api.func @function_identity(%input: !tvm_ffi_abi.function) -> !tvm_ffi_abi.function {
  %any = tvm_ffi_api.upcast %input : !tvm_ffi_abi.function -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// A dispatcher forwards a borrowed argument pack to typed packed functions. Failed ABI
// calls propagate; only successful Exception payloads trigger the next attempt.
// CHECK-LABEL: llvm.func @__tvm_ffi_dispatch(
// CHECK-NOT: llvm.call @__tvm_ffi_identity
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: llvm.return
tvm_ffi_api.pack.func @dispatch(%input: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_api.any {
  %ok, %value = tvm_ffi_api.pack.call @identity(%input) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  %ready = tvm_ffi_api.get %ok : !tvm_ffi_abi.bool -> i1
  %success, %payload = scf.if %ready -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
    %rejected = tvm_ffi_api.isa %value is !tvm_ffi_api.exception : !tvm_ffi_api.any
    %retry_ok, %retry_value = scf.if %rejected -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
      %next_ok, %next_value = tvm_ffi_api.pack.call @identity(%input) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
      scf.yield %next_ok, %next_value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
    } else {
      scf.yield %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
    }
    scf.yield %retry_ok, %retry_value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  } else {
    scf.yield %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
  }
  tvm_ffi_api.return %success, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
