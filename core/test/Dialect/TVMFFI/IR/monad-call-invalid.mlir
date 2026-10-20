// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @missing() {
  // expected-error@+1 {{callee must resolve to a tvm_ffi.func}}
  %result = tvm_ffi.monad.call @absent() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

tvm_ffi.func @callee(!tvm_ffi.int) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_arguments(%input: !tvm_ffi.bool) {
  // expected-error@+1 {{operand types must match the callee arguments}}
  %result = tvm_ffi.monad.call @callee(%input) : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_payload() {
  // expected-error@+1 {{Result payload must match the callee result}}
  %result = tvm_ffi.monad.call @callee() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.tensor
func.func @missing_wrapper() {
  // expected-error@+1 {{callee must have emit_tvm_ffi_abi}}
  %result = tvm_ffi.monad.call @callee() : () -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return
}
