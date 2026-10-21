// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @missing() {
  %result = tvm_ffi.monad.call @absent() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

tvm_ffi.func @callee(!tvm_ffi.int) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_arguments(%input: !tvm_ffi.bool) {
  %result = tvm_ffi.monad.call @callee(%input) : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_payload() {
  // expected-error@+1 {{Result payload must match the callee result}}
  %result = tvm_ffi.monad.call @callee() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.c.tensor
func.func @missing_wrapper() {
  %result = tvm_ffi.monad.call @callee() : () -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return
}
