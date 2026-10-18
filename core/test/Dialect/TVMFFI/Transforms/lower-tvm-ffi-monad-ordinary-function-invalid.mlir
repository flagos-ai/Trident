// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-diagnostics

// expected-error@+1 {{inline ordinary functions carrying Results before running lower-tvm-ffi-monad}}
func.func private @helper(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  return %input : !tvm_ffi.monad.result<i64>
}
