// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @not_result(%input: i64) {
  // expected-error@+1 {{operand #0 must be}}
  %success, %value = tvm_ffi.monad.unpack %input : (i64) -> (i1, i64)
  return
}

// -----

func.func @wrong_payload(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  // expected-error@+2 {{inferred type}}
  // expected-error@+1 {{failed to infer returned types}}
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.int)
  return
}

// -----

func.func @missing_widening(%input: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
  // expected-error@+2 {{inferred type}}
  // expected-error@+1 {{failed to infer returned types}}
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> (i1, !tvm_ffi.c.tensor)
  return
}

