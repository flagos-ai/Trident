// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{union member types must be unique}}
func.func @duplicate_union_member(
    %arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.int>)

// -----

// expected-error@+1 {{expected '<'}}
func.func private @old_array(!tvm_ffi.array)

// -----

// expected-error@+1 {{union requires at least two member types}}
func.func @single_member_union(%arg: !tvm_ffi.union<!tvm_ffi.int>)

// -----

// expected-error@+1 {{invalid union member type '!tvm_ffi.c.any'}}
func.func private @union_c_any(!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.any>)

// -----

// expected-error@+1 {{union members must be TVM FFI C ABI types}}
func.func private @union_native_integer(!tvm_ffi.union<!tvm_ffi.int, i64>)

// -----

// expected-error@+1 {{unknown  type `c.unknown` in dialect `tvm_ffi`}}
func.func private @unknown(!tvm_ffi.c.unknown)
