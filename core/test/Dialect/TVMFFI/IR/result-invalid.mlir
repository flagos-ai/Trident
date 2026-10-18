// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

tvm_ffi.func @failure_with_payload(%value: i64) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires zero operands}}
  %result = "tvm_ffi.monad.fail"(%value) : (i64) -> !tvm_ffi.monad.result<i64>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

tvm_ffi.func @missing_payload(%ready: i1) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{payload count must match Result count}}
  %value = tvm_ffi.monad.pure : () -> !tvm_ffi.monad.result<i64>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// Missing terminators are a dialect verifier contract, independent of Python.
tvm_ffi.func @missing_yield(%input: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{body must end with tvm_ffi.monad.yield}}
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb_unit(%unit_payload: !tvm_ffi.none):
    %unused = arith.constant 7 : i64
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

tvm_ffi.func @native_bind_yield(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%payload: i64):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi.monad.yield %payload : i64
  }

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// expected-error@+1 {{nested Result payloads are unsupported}}
tvm_ffi.func @nested(%value: !tvm_ffi.monad.result<!tvm_ffi.monad.result<i64>>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

tvm_ffi.func @wrong_argument(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%payload: i32):
    tvm_ffi.monad.yield %input : !tvm_ffi.monad.result<i64>
  }

  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}

// -----

tvm_ffi.func @wrong_parent() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{must be nested directly in tvm_ffi.monad.bind}}
  tvm_ffi.monad.yield
}

// -----

tvm_ffi.func @wrong_pure_payload(%value: i32) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{payload type must match Result type}}
  %result = tvm_ffi.monad.pure %value : (i32) -> !tvm_ffi.monad.result<i64>

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

tvm_ffi.func @wrong_yield(%input: !tvm_ffi.monad.result<i64>, %other: !tvm_ffi.monad.result<i32>) -> !tvm_ffi.monad.result<i64> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%payload: i64):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi.monad.yield %other : !tvm_ffi.monad.result<i32>
  }

  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}

// -----

tvm_ffi.func @zero_fail_results() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.fail"() : () -> ()

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

tvm_ffi.func @zero_pure_results() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.pure"() : () -> ()

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
tvm_ffi.func @empty_payload() -> !tvm_ffi.monad.result<>

// -----

tvm_ffi.func @omitted_none_argument(%unit: !tvm_ffi.monad.result<!tvm_ffi.none>, %input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  %output = tvm_ffi.monad.bind %unit, %input : (!tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%value: i64):
    tvm_ffi.monad.yield %input : !tvm_ffi.monad.result<i64>
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}
