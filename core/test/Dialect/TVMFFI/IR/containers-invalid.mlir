// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @array_create(%value: !tvm_ffi.int) {
  // expected-error@+1 {{elements must match the array element type}}
  %result = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.bool>
  return
}

// -----

func.func @array_create_payload(%value: !tvm_ffi.int) {
  // expected-error@+1 {{result #0 must be An array whose elements are TVM FFI C ABI values, but got '!tvm_ffi.int'}}
  %result = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.int
  return
}

// -----

func.func @array_get(%array: !tvm_ffi.array<!tvm_ffi.int>, %index: i64) {
  // expected-error@+1 {{result must match array element type}}
  %result = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.bool
  return
}

// -----

func.func @array_size(%array: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{expected tvm_ffi.int}}
  %result = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.bool
  return
}

// -----

// expected-error@+1 {{array element must be a TVM FFI C ABI type}}
func.func private @array_native(!tvm_ffi.array<i64>)

// -----

// expected-error@+1 {{array element must be a TVM FFI C ABI type}}
func.func private @array_object(!tvm_ffi.array<!tvm_ffi.object>)

// -----

// expected-error@+1 {{array element must be a TVM FFI C ABI type}}
func.func private @array_result(!tvm_ffi.array<!tvm_ffi.monad.result<!tvm_ffi.int>>)

// -----

// expected-error@+1 {{array element must be a TVM FFI C ABI type}}
func.func private @array_tensor(!tvm_ffi.array<tensor<2xi64>>)

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @result(!tvm_ffi.tuple<!tvm_ffi.monad.result<!tvm_ffi.int>>)

// -----

func.func @tuple_construct_count(%value: !tvm_ffi.int) {
  // expected-error@+1 {{elements must match the tuple types}}
  %result = tvm_ffi.tuple.construct(%value) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.int>
  return
}

// -----

func.func @tuple_construct_type(%value: !tvm_ffi.int) {
  // expected-error@+1 {{elements must match the tuple types}}
  %result = tvm_ffi.tuple.construct(%value) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool>
  return
}

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @tuple_native(!tvm_ffi.tuple<!tvm_ffi.int, i64>)

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @tuple_object(!tvm_ffi.tuple<!tvm_ffi.object>)

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @tuple_tensor(!tvm_ffi.tuple<tensor<2xi64>>)

// -----

func.func @tuple_unconstruct_count(%tuple: !tvm_ffi.tuple<!tvm_ffi.int>) {
  // expected-error@+1 {{results must match the tuple types}}
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.int>) -> ()
  return
}

// -----

func.func @tuple_unconstruct_type(%tuple: !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>) {
  // expected-error@+1 {{results must match the tuple types}}
  %first, %second = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>) -> (!tvm_ffi.bool, !tvm_ffi.int)
  return
}
