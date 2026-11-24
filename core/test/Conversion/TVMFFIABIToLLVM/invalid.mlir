// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-abi-to-llvm --verify-diagnostics

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @undecomposed(%input: !tvm_ffi_api.array<!tvm_ffi_abi.int>) {
  %ok, %value = tvm_ffi_api.call @callee(%input) : (!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return
}

tvm_ffi_api.func @callee(!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.none

// -----

// LLVM conversion rejects Result signatures without monad decomposition.
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @unlowered_bind(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  %result = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%value: !tvm_ffi_abi.int):
    %pure = tvm_ffi_monad.pure %value : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    tvm_ffi_monad.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }
  return %result : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}
