// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{requires exactly one pack.arguments input}}
tvm_ffi_api.pack.func @empty() -> !tvm_ffi_abi.none

// -----

// expected-error @+1 {{requires exactly one pack.arguments input}}
tvm_ffi_api.pack.func @typed(%arg: !tvm_ffi_abi.int) -> !tvm_ffi_abi.none

// -----

// expected-error @+1 {{function must return exactly one C ABI value}}
tvm_ffi_api.pack.func @escape(%arg: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.pack.arguments

// -----

// expected-error @+1 {{pack.arguments inputs require pack.func}}
tvm_ffi_api.func @typed(%arg: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.none

// -----

func.func @ordinary(%arg: !tvm_ffi_api.any) {
  // expected-error @+1 {{operand types must match the callee arguments}}
  %ok, %value = tvm_ffi_api.call @typed(%arg) : (!tvm_ffi_api.any) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return
}
tvm_ffi_api.func @typed(%arg: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
