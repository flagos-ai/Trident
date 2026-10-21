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

// CHECK-LABEL: llvm.func @identity(
// CHECK-LABEL: llvm.func @__tvm_ffi_identity(
// CHECK: llvm.call @identity
// CHECK-LABEL: llvm.func @function_identity(
// CHECK: llvm.ptrtoint
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @__tvm_ffi_function_identity(
// CHECK: llvm.inttoptr
// CHECK: llvm.call @function_identity
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
