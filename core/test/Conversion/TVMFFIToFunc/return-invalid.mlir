// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-func --verify-diagnostics

// expected-note@+1 {{see existing symbol definition here}}
func.func private @__tvm_ffi_collision(i64)

// expected-error@+1 {{redefinition of symbol named '__tvm_ffi_collision'}}
tvm_ffi.pack.func @collision(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}

// -----

// Conversion preserves public visibility instead of making declarations private.
// expected-error@+1 {{symbol declaration cannot have public visibility}}
tvm_ffi.pack.func public @external(!tvm_ffi.pack.arguments) -> !tvm_ffi.none
