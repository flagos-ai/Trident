// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @empty()

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @monadic() -> !tvm_ffi.monad.result<!tvm_ffi.int>

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @multiple() -> (!tvm_ffi.int, !tvm_ffi.int)

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @native() -> i64

// -----

tvm_ffi.func @wrong_flag(%flag: i1, %payload: !tvm_ffi.any) -> !tvm_ffi.int {
  // expected-error@+1 {{operand #0 must be}}
  "tvm_ffi.return"(%flag, %payload) : (i1, !tvm_ffi.any) -> ()
}

// -----

tvm_ffi.func @wrong_payload(%flag: !tvm_ffi.bool, %payload: !tvm_ffi.int) -> !tvm_ffi.int {
  // expected-error@+1 {{operand #1 must be}}
  "tvm_ffi.return"(%flag, %payload) : (!tvm_ffi.bool, !tvm_ffi.int) -> ()
}

// -----

tvm_ffi.func @missing_flag(%payload: !tvm_ffi.any) -> !tvm_ffi.int {
  // expected-error@+1 {{expected 2 operands}}
  "tvm_ffi.return"(%payload) : (!tvm_ffi.any) -> ()
}
