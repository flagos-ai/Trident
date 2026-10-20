// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @missing() {
  // expected-error@+1 {{callee must resolve to a tvm_ffi.func}}
  %ok, %value = tvm_ffi.call @absent() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

tvm_ffi.func @callee(!tvm_ffi.int) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_arguments(%input: !tvm_ffi.bool) {
  // expected-error@+1 {{operand types must match the callee arguments}}
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.bool) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi}
func.func @wrong_payload() {
  // expected-error@+1 {{result #1 must be}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.bool, !tvm_ffi.int)
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.tensor
func.func @missing_wrapper() {
  // expected-error@+1 {{callee must have emit_tvm_ffi_abi}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}
func.func @missing_payload() {
  // expected-error@+1 {{expected 2 results}}
  %ok = "tvm_ffi.call"() {callee = @callee} : () -> !tvm_ffi.bool
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}
func.func @wrong_flag() {
  // expected-error@+1 {{result #0 must be}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.int, !tvm_ffi.any)
  return
}

// -----

func.func private @callee(!tvm_ffi.int, !llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi}
func.func @wrong_lowered_arguments(%input: !tvm_ffi.bool) {
  // expected-error@+1 {{operand types must match the lowered callee arguments}}
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.bool) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

func.func private @callee(!llvm.ptr) -> i32
func.func @missing_lowered_marker() {
  // expected-error@+1 {{callee must have emit_tvm_ffi_abi}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}
