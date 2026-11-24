// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-abi-to-func --verify-diagnostics

// expected-note@+1 {{see existing symbol definition here}}
func.func private @__tvm_ffi_collision(i64)

// expected-error@+1 {{redefinition of symbol named '__tvm_ffi_collision'}}
tvm_ffi_abi.pack.func @collision(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.none {
  %none = tvm_ffi_abi.constant.none
  %any = tvm_ffi_abi.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}

// -----

// Conversion preserves public visibility instead of making declarations private.
// expected-error@+1 {{symbol declaration cannot have public visibility}}
tvm_ffi_abi.pack.func public @external(!tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.none
