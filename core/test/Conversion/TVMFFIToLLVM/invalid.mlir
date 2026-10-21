// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-llvm --verify-diagnostics

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @undecomposed(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

tvm_ffi.func @callee(!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}

// -----

// LLVM conversion rejects Result signatures without monad decomposition.
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @unlowered_bind(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%value: !tvm_ffi.int):
    %pure = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
}
