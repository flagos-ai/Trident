// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @invalid_erased(%a: !tvm_ffi_api.any, %b: !tvm_ffi_abi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool, Float or Int}}
  %result = tvm_ffi_arith.sub %a, %b : (!tvm_ffi_api.any, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
  return
}

// -----

func.func @invalid_integral(%a: !tvm_ffi_abi.float, %b: !tvm_ffi_abi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool or Int}}
  %result = tvm_ffi_arith.floordiv %a, %b : (!tvm_ffi_abi.float, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
  return
}

// -----

func.func @invalid_promotion(%a: !tvm_ffi_abi.bool, %b: !tvm_ffi_abi.bool) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) '!tvm_ffi_abi.int' are incompatible with return type(s) of operation '!tvm_ffi_abi.bool'}}
  %result = tvm_ffi_arith.add %a, %b : (!tvm_ffi_abi.bool, !tvm_ffi_abi.bool) -> !tvm_ffi_abi.bool
  return
}

// -----

func.func @invalid_native(%a: i64, %b: !tvm_ffi_abi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI Bool, Float or Int}}
  %result = tvm_ffi_arith.add %a, %b : (i64, !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
  return
}
