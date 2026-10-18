// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-diagnostics

tvm_ffi.func @undecomposed(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{failed to legalize operation 'tvm_ffi.monad.bind' that was explicitly marked illegal}}
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%value: i64):
    %pure = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<i64>
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}
