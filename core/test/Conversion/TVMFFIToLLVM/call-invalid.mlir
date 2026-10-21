// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-diagnostics

// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @undecomposed(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

tvm_ffi.func @callee(!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}
