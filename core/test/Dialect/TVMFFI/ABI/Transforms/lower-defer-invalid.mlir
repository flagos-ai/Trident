// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --allow-unregistered-dialect --lower-tvm-ffi-abi-defer --split-input-file --verify-diagnostics

// Module bodies are graph regions, even when a terminator is present.
module {
  // expected-error@+1 {{requires an SSACFG region for deferred cleanup}}
  tvm_ffi_abi.defer { tvm_ffi_abi.defer_yield }
  llvm.unreachable
}

// -----

// Unknown single-block regions do not establish SSA dominance semantics.
"test.scope"() ({
  // expected-error@+1 {{requires an SSACFG region for deferred cleanup}}
  tvm_ffi_abi.defer { tvm_ffi_abi.defer_yield }
  llvm.unreachable
}) : () -> ()

// -----

// Multi-block regions have SSA dominance, but an unknown final operation does
// not establish a terminator. Validate downstream blocks before changing IR.
// expected-error@+1 {{requires a block terminator for deferred cleanup}}
"test.scope"() ({
  tvm_ffi_abi.defer { tvm_ffi_abi.defer_yield }
  cf.br ^exit
^exit:
  "test.no_terminator"() : () -> ()
}) : () -> ()
