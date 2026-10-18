// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

// expected-error@+1 {{function results must be nonempty monadic Result types}}
tvm_ffi.func @empty()

// -----

tvm_ffi.func @empty_return() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{Result types must match the function results}}
  tvm_ffi.return
}

// -----

tvm_ffi.func @missing_payload(%input: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{Result types must match the function results}}
  tvm_ffi.return %input : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// expected-error@+1 {{function results must be nonempty monadic Result types}}
tvm_ffi.func @mixed() -> (!tvm_ffi.monad.result<i64>, i64)

// -----

tvm_ffi.func @native_operand(%input: i64) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{operand #0 must be variadic of One payload guarded by a success bit}}
  tvm_ffi.return %input : i64
}

// -----

// expected-error@+1 {{function results must be nonempty monadic Result types}}
tvm_ffi.func @ordinary() -> i64

// -----

tvm_ffi.func @wrong_count(%input: !tvm_ffi.monad.result<i64>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  // expected-error@+1 {{Result types must match the function results}}
  tvm_ffi.return %input : !tvm_ffi.monad.result<i64>
}

// -----

tvm_ffi.func @wrong_type(%input: !tvm_ffi.monad.result<i32>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{Result types must match the function results}}
  tvm_ffi.return %input : !tvm_ffi.monad.result<i32>
}
