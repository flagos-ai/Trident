// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @cast_rejects_concrete_result(%arg: !tvm_ffi_abi.float) {
  // expected-error@+1 {{cast incompatible}}
  %result = "tvm_ffi_api.upcast"(%arg)
     : (!tvm_ffi_abi.float) -> !tvm_ffi_abi.int
  return
}

// -----

func.func @cast_rejects_incompatible_union(%arg: !tvm_ffi_abi.int) {
  // expected-error@+1 {{'tvm_ffi_api.upcast' op operand type '!tvm_ffi_abi.int' and result type '!tvm_ffi_api.union<!tvm_ffi_abi.float, !tvm_ffi_abi.exception>' are cast incompatible}}
  %result = tvm_ffi_api.upcast %arg
     : !tvm_ffi_abi.int
      -> !tvm_ffi_api.union<!tvm_ffi_abi.float, !tvm_ffi_abi.exception>
  return
}

// -----

func.func @cast_rejects_native_operand(%arg: i64) {
  // expected-error@+1 {{'tvm_ffi_api.upcast' op operand #0 must be TVM FFI value with a TVMFFIAny ABI representation, but got 'i64'}}
  %result = "tvm_ffi_api.upcast"(%arg)
     : (i64) -> !tvm_ffi_api.any
  return
}

// -----

func.func @downcast_concrete_source(%arg: !tvm_ffi_abi.int) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi_api.downcast %arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.float
  return
}

// -----

func.func @downcast_not_subset(%arg: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi_api.downcast %arg : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
  return
}

// -----

func.func @downcast_object(%arg: !tvm_ffi_api.any) {
  // expected-error@+1 {{result #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = tvm_ffi_api.downcast %arg : !tvm_ffi_api.any -> !tvm_ffi_abi.object
  return
}

// -----

func.func @downcast_same_union(%arg: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi_api.downcast %arg : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.none, !tvm_ffi_abi.int>
  return
}

// -----

func.func @upcast_object(%arg: !tvm_ffi_abi.object) {
  // expected-error@+1 {{operand #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = tvm_ffi_api.upcast %arg : !tvm_ffi_abi.object -> !tvm_ffi_api.any
  return
}
