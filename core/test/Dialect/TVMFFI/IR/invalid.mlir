//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @cast_rejects_concrete_result(%arg: !tvm_ffi.float) {
  // expected-error@+1 {{cast incompatible}}
  %result = "tvm_ffi.upcast"(%arg)
      : (!tvm_ffi.float) -> !tvm_ffi.int
  return
}

// -----

func.func @cast_rejects_incompatible_union(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.upcast' op operand type '!tvm_ffi.int' and result type '!tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.exception>' are cast incompatible}}
  %result = tvm_ffi.upcast %arg
      : !tvm_ffi.int
      -> !tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.exception>
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

// expected-error@+2 {{union member types must be unique}}
func.func @duplicate_union_member(
    %arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.int>)

// -----

func.func @empty_any_return() -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  // expected-error@+1 {{has 0 operands, but enclosing function (@empty_any_return) returns 1}}
  return
}

// -----

func.func @get_int_width(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.int' to 'i128'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.int) -> i128
  return
}

// -----

func.func @get_unsupported_array(%arg: !tvm_ffi.c.array) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.c.array' to 'i64'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.c.array) -> i64
  return
}

// -----

func.func @invalid_dtype_constant() {
  // expected-error@+1 {{dtype result requires [code, bits, lanes]}}
  %value = "tvm_ffi.constant.dtype"() <{value = [1, 2]}> : () -> !tvm_ffi.dtype
  return
}

// -----

func.func @invalid_union_return() -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool>> {
  %value = "tvm_ffi.constant.float"() <{value = 1.0 : f64}>
      : () -> !tvm_ffi.float
  // expected-error@+1 {{are cast incompatible}}
  %return_cast_0 = tvm_ffi.upcast %value : !tvm_ffi.float -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool>
  %returned = tvm_ffi.monad.pure %return_cast_0 : (!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool>>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool>>
}

// -----

// expected-error@+1 {{must have public visibility}}
tvm_ffi.func private @private_function() -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %value = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}

// -----

// expected-error@+1 {{union requires at least two member types}}
func.func @single_member_union(%arg: !tvm_ffi.union<!tvm_ffi.int>)

// -----

func.func @to_mismatched_native_type(%value: i64) {
  // expected-error@+1 {{'tvm_ffi.to' op unsupported native-to-TVM FFI conversion from 'i64' to '!tvm_ffi.bool'}}
  %result = tvm_ffi.to %value : i64 -> !tvm_ffi.bool
  return
}
