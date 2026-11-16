// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @invalid_erased(%a: !tvm_ffi.any, %b: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool, Float or Int}}
  %result = tvm_ffi.arith.sub %a, %b : (!tvm_ffi.any, !tvm_ffi.int) -> !tvm_ffi.int
  return
}

// -----

func.func @invalid_integral(%a: !tvm_ffi.float, %b: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool or Int}}
  %result = tvm_ffi.arith.floordiv %a, %b : (!tvm_ffi.float, !tvm_ffi.int) -> !tvm_ffi.int
  return
}

// -----

func.func @invalid_promotion(%a: !tvm_ffi.bool, %b: !tvm_ffi.bool) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) '!tvm_ffi.int' are incompatible with return type(s) of operation '!tvm_ffi.bool'}}
  %result = tvm_ffi.arith.add %a, %b : (!tvm_ffi.bool, !tvm_ffi.bool) -> !tvm_ffi.bool
  return
}

// -----

func.func @invalid_native(%a: i64, %b: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool, Float or Int}}
  %result = tvm_ffi.arith.add %a, %b : (i64, !tvm_ffi.int) -> !tvm_ffi.int
  return
}
