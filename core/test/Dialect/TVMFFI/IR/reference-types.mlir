// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

// HasObject and Contains satisfy the combined constraint. Native Object is
// admitted separately, without acquiring C ABI object classification.
func.func @accepted(%any: !tvm_ffi.any, %array: !tvm_ffi.c.array, %exception: !tvm_ffi.exception, %function: !tvm_ffi.function, %object: !tvm_ffi.object, %str: !tvm_ffi.str, %tensor: !tvm_ffi.tensor, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  tvm_ffi.ObjectIncRef %any : !tvm_ffi.any
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.c.array
  tvm_ffi.ObjectIncRef %exception : !tvm_ffi.exception
  tvm_ffi.ObjectDecRef %function : !tvm_ffi.function
  tvm_ffi.ObjectIncRef %object : !tvm_ffi.object
  tvm_ffi.ObjectDecRef %str : !tvm_ffi.str
  tvm_ffi.ObjectIncRef %tensor : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %union : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return
}

// -----

func.func @array(%value: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.array<!tvm_ffi.int>
  return
}

// -----

func.func @result(%value: !tvm_ffi.monad.result<!tvm_ffi.tensor>) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return
}

// -----

func.func @scalar(%value: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.int
  return
}

// -----

func.func @tuple(%value: !tvm_ffi.tuple<>) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.tuple<>
  return
}
