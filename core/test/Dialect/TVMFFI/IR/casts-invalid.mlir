// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @cast_rejects_concrete_result(%arg: !tvm_ffi.float) {
  // expected-error@+1 {{cast incompatible}}
  %result = "tvm_ffi.upcast"(%arg)
     : (!tvm_ffi.float) -> !tvm_ffi.int
  return
}

// -----

func.func @cast_rejects_incompatible_union(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.upcast' op operand type '!tvm_ffi.int' and result type '!tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.c.exception>' are cast incompatible}}
  %result = tvm_ffi.upcast %arg
     : !tvm_ffi.int
      -> !tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.c.exception>
  return
}

// -----

func.func @cast_rejects_native_operand(%arg: i64) {
  // expected-error@+1 {{'tvm_ffi.upcast' op operand #0 must be TVM FFI value with a TVMFFIAny ABI representation, but got 'i64'}}
  %result = "tvm_ffi.upcast"(%arg)
     : (i64) -> !tvm_ffi.any
  return
}

// -----

func.func @downcast_concrete_source(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.int -> !tvm_ffi.float
  return
}

// -----

func.func @downcast_not_subset(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return
}

// -----

func.func @downcast_object(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{result #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.object
  return
}

// -----

func.func @downcast_same_union(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.none, !tvm_ffi.int>
  return
}

// -----

func.func @upcast_object(%arg: !tvm_ffi.object) {
  // expected-error@+1 {{operand #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = tvm_ffi.upcast %arg : !tvm_ffi.object -> !tvm_ffi.any
  return
}
