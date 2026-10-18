// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --allow-unregistered-dialect --lower-tvm-ffi-monad --verify-diagnostics

tvm_ffi.func @unsupported(%value: i64, %ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %input = scf.if %ready -> (!tvm_ffi.monad.result<i64>) {
    %input_pure = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    scf.yield %input_pure : !tvm_ffi.monad.result<i64>
  } else {
    %input_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
    scf.yield %input_fail : !tvm_ffi.monad.result<i64>
  }
  // expected-error@+1 {{failed to legalize operation 'test.consume' that was explicitly marked illegal}}
  "test.consume"(%input) : (!tvm_ffi.monad.result<i64>) -> ()

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}
