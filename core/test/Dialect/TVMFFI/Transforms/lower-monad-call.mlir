// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.transfer' --implicit-check-not='tvm_ffi.monad.' --implicit-check-not='llvm.call' --implicit-check-not='unrealized_conversion_cast'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[INPUT]]) {arg_attrs = [{test.argument}], res_attrs = [{}, {test.result}]}
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[OK]], %[[VALUE]])
// CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// LLVM-LABEL: llvm.func @invoke(
// LLVM: llvm.call @__tvm_ffi_identity
// LLVM: llvm.return
func.func @invoke(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  %result = tvm_ffi.monad.call @identity(%input) {arg_attrs = [{test.argument}], res_attrs = [{test.result}]} : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
}

// CHECK-LABEL: func.func @invoke_function(
// CHECK: %[[FUNCTION_OK:[a-zA-Z0-9_]+]], %[[FUNCTION_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @function_identity(%[[INPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[FUNCTION_OK]], %[[FUNCTION_VALUE]])
// CHECK-NEXT: return %[[RESULT]]
// LLVM-LABEL: llvm.func @invoke_function(
// LLVM: llvm.ptrtoint
// LLVM: llvm.call @__tvm_ffi_function_identity
// LLVM: llvm.return
func.func @invoke_function(%input: !tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function> {
  %result = tvm_ffi.monad.call @function_identity(%input) : (!tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.function>
}

// CHECK-LABEL: func.func @invoke_unmarked(
// CHECK: %[[UNMARKED_OK:[a-zA-Z0-9_]+]], %[[UNMARKED_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[UNMARKED_INPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT: %[[UNMARKED_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[UNMARKED_OK]], %[[UNMARKED_VALUE]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK-NEXT: return %[[UNMARKED_RESULT]]
// LLVM-LABEL: llvm.func @invoke_unmarked(
// LLVM: llvm.call @__tvm_ffi_identity
// LLVM: llvm.return
func.func @invoke_unmarked(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  %result = tvm_ffi.monad.call @identity(%input) {res_attrs = []} : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
}

// LLVM-LABEL: llvm.func @identity(
// LLVM-LABEL: llvm.func @__tvm_ffi_identity(
// LLVM: llvm.call @identity
// LLVM-LABEL: llvm.func @function_identity(
// LLVM: llvm.ptrtoint
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_function_identity(
// LLVM: llvm.inttoptr
// LLVM: llvm.call @function_identity
// LLVM: llvm.return
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
