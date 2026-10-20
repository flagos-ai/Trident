// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-diagnostics

func.func @undecomposed(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+2 {{requires decompose-tvm-ffi before LLVM conversion}}
  // expected-error@+1 {{failed to legalize operation 'tvm_ffi.call'}}
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

tvm_ffi.func @callee(!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}
