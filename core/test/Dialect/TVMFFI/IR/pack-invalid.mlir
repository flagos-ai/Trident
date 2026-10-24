// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{requires exactly one pack.arguments input}}
tvm_ffi.pack.func @empty() -> !tvm_ffi.none

// -----

// expected-error @+1 {{requires exactly one pack.arguments input}}
tvm_ffi.pack.func @typed(%arg: !tvm_ffi.int) -> !tvm_ffi.none

// -----

// expected-error @+1 {{function must return exactly one C ABI value}}
tvm_ffi.pack.func @escape(%arg: !tvm_ffi.pack.arguments) -> !tvm_ffi.pack.arguments

// -----

// expected-error @+1 {{pack.arguments inputs require pack.func}}
tvm_ffi.func @typed(%arg: !tvm_ffi.pack.arguments) -> !tvm_ffi.none

// -----

func.func @ordinary(%arg: !tvm_ffi.any) {
  // expected-error @+1 {{operand types must match the callee arguments}}
  %ok, %value = tvm_ffi.call @typed(%arg) : (!tvm_ffi.any) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}
tvm_ffi.func @typed(%arg: !tvm_ffi.int) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi}
