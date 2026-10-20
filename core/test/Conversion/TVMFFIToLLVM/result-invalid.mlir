// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-diagnostics

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
