// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @downcast_any_result(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.any
  return
}

// -----

func.func @downcast_concrete_source(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be widened TVM FFI any or union value}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.int -> !tvm_ffi.float
  return
}

// -----

func.func @downcast_unrelated(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.float
  return
}

// -----

func.func @downcast_not_subset(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return
}

// -----

func.func @downcast_same_union(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.none, !tvm_ffi.int>
  return
}

// -----

func.func @downcast_function(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.function
  return
}

// -----

func.func @downcast_object(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.object
  return
}

// -----

func.func @downcast_function_union(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.downcast %arg : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>
  return
}

// -----

func.func @upcast_function(%arg: !tvm_ffi.function) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.upcast %arg : !tvm_ffi.function -> !tvm_ffi.any
  return
}

// -----

func.func @upcast_object(%arg: !tvm_ffi.object) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.upcast %arg : !tvm_ffi.object -> !tvm_ffi.any
  return
}

// -----

func.func @upcast_function_union(%arg: !tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>) {
  // expected-error@+1 {{cast incompatible}}
  %result = tvm_ffi.upcast %arg : !tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none> -> !tvm_ffi.any
  return
}

// -----

func.func @downcast_native_result(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{result #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = "tvm_ffi.downcast"(%arg) : (!tvm_ffi.any) -> i64
  return
}

// -----

func.func @downcast_result_payload(%arg: !tvm_ffi.any) {
  // expected-error@+1 {{result #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = "tvm_ffi.downcast"(%arg) : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

func.func @upcast_result_operand(%arg: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  // expected-error@+1 {{operand #0 must be TVM FFI value with a TVMFFIAny ABI representation}}
  %result = "tvm_ffi.upcast"(%arg) : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.any
  return
}

