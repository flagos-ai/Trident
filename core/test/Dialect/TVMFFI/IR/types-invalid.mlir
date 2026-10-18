// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{unknown  type `c.any` in dialect `tvm_ffi`}}
func.func private @old_any(!tvm_ffi.c.any)

// -----

// expected-error@+1 {{expected '<'}}
func.func private @old_array(!tvm_ffi.array)

// -----

// expected-error@+1 {{unknown  type `c.bool` in dialect `tvm_ffi`}}
func.func private @old_bool(!tvm_ffi.c.bool)

// -----

// expected-error@+1 {{unknown  type `c.device` in dialect `tvm_ffi`}}
func.func private @old_device(!tvm_ffi.c.device)

// -----

// expected-error@+1 {{unknown  type `c.dtype` in dialect `tvm_ffi`}}
func.func private @old_dtype(!tvm_ffi.c.dtype)

// -----

// expected-error@+1 {{unknown  type `c.float` in dialect `tvm_ffi`}}
func.func private @old_float(!tvm_ffi.c.float)

// -----

// expected-error@+1 {{unknown  type `c.function` in dialect `tvm_ffi`}}
func.func private @old_function(!tvm_ffi.c.function)

// -----

// expected-error@+1 {{unknown  type `c.int` in dialect `tvm_ffi`}}
func.func private @old_int(!tvm_ffi.c.int)

// -----

// expected-error@+1 {{unknown  type `c.none` in dialect `tvm_ffi`}}
func.func private @old_none(!tvm_ffi.c.none)

// -----

// expected-error@+1 {{unknown  type `c.object` in dialect `tvm_ffi`}}
func.func private @old_object(!tvm_ffi.c.object)

// -----

// expected-error@+1 {{unknown  type `c.raw_str` in dialect `tvm_ffi`}}
func.func private @old_raw_str(!tvm_ffi.c.raw_str)

// -----

// expected-error@+1 {{unknown  type `c.small_str` in dialect `tvm_ffi`}}
func.func private @old_small_str(!tvm_ffi.c.small_str)

// -----

// expected-error@+1 {{unknown  type `c.str` in dialect `tvm_ffi`}}
func.func private @old_str(!tvm_ffi.c.str)

// -----

// expected-error@+1 {{unknown  type `c.tensor` in dialect `tvm_ffi`}}
func.func private @old_tensor(!tvm_ffi.c.tensor)
