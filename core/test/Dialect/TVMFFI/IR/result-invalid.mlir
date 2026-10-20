// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @failure_with_payload(%value: i64) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires zero operands}}
  %result = "tvm_ffi.monad.fail"(%value) : (i64) -> !tvm_ffi.monad.result<!tvm_ffi.int>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

func.func @missing_payload(%ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{payload count must match Result count}}
  %value = tvm_ffi.monad.pure : () -> !tvm_ffi.monad.result<!tvm_ffi.int>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// Missing terminators are a dialect verifier contract, independent of Python.
func.func @missing_yield(%input: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{body must end with tvm_ffi.monad.yield}}
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb_unit(%unit_payload: !tvm_ffi.none):
    %unused = arith.constant 7 : i64
  }
  return %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

func.func @payload_bind_yield(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%payload: !tvm_ffi.int):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi.monad.yield %payload : !tvm_ffi.int
  }

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func @nested(%value: !tvm_ffi.monad.result<!tvm_ffi.monad.result<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

func.func @wrong_argument(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%payload: i32):
    tvm_ffi.monad.yield %input : !tvm_ffi.monad.result<!tvm_ffi.int>
  }

  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

func.func @wrong_parent() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{must be nested directly in tvm_ffi.monad.bind}}
  tvm_ffi.monad.yield
}

// -----

func.func @wrong_pure_payload(%value: i32) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{payload type must match Result type}}
  %result = tvm_ffi.monad.pure %value : (i32) -> !tvm_ffi.monad.result<!tvm_ffi.int>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

func.func @wrong_yield(%input: !tvm_ffi.monad.result<!tvm_ffi.int>, %other: !tvm_ffi.monad.result<!tvm_ffi.bool>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%payload: !tvm_ffi.int):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi.monad.yield %other : !tvm_ffi.monad.result<!tvm_ffi.bool>
  }

  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

func.func @zero_fail_results() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.fail"() : () -> ()

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

func.func @zero_pure_results() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.pure"() : () -> ()

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @empty_payload() -> !tvm_ffi.monad.result<>

// -----

func.func @omitted_none_argument(%unit: !tvm_ffi.monad.result<!tvm_ffi.none>, %input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  %output = tvm_ffi.monad.bind %unit, %input : (!tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%value: !tvm_ffi.int):
    tvm_ffi.monad.yield %input : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

func.func @zero_input_wrong_arguments() {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  tvm_ffi.monad.bind : () -> () {
  ^bb0(%unexpected: i64):
    tvm_ffi.monad.yield
  }
  return
}

// -----

func.func @zero_output_wrong_yield(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%payload: !tvm_ffi.int):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi.monad.yield %input : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_float(!tvm_ffi.monad.result<f64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_index(!tvm_ffi.monad.result<index>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_integer(!tvm_ffi.monad.result<i64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_object(!tvm_ffi.monad.result<!tvm_ffi.object>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_pointer(!tvm_ffi.monad.result<!llvm.ptr>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_tensor(!tvm_ffi.monad.result<tensor<2xi64>>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_vector(!tvm_ffi.monad.result<vector<2xi32>>)
