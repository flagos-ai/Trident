// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-diagnostics

// Decomposition owns continuation control flow; LLVM conversion requires it.
func.func @unlowered_bind(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{failed to legalize operation 'tvm_ffi.monad.bind'}}
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%value: i64):
    %pure = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<i64>
  }
  return %result : !tvm_ffi.monad.result<i64>
}
