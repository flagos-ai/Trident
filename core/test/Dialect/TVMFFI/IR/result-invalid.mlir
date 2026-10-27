// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @missing() {
  %ok, %value = tvm_ffi.call @absent() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

tvm_ffi.func @callee(!tvm_ffi.int) -> !tvm_ffi.c.tensor
func.func @wrong_arguments(%input: !tvm_ffi.bool) {
  // expected-error@+1 {{operand types must match the callee arguments}}
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.bool) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.c.tensor
func.func @wrong_payload() {
  // expected-error@+1 {{result #1 must be}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.bool, !tvm_ffi.int)
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.none
func.func @missing_payload() {
  // expected-error@+1 {{expected 2 results}}
  %ok = "tvm_ffi.call"() {callee = @callee} : () -> !tvm_ffi.bool
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.none
func.func @wrong_flag() {
  // expected-error@+1 {{result #0 must be}}
  %ok, %value = tvm_ffi.call @callee() : () -> (!tvm_ffi.int, !tvm_ffi.any)
  return
}

// -----

func.func private @callee(!tvm_ffi.int, !llvm.ptr) -> i32
func.func @wrong_lowered_arguments(%input: !tvm_ffi.bool) {
  // expected-error@+1 {{operand types must match the lowered callee arguments}}
  %ok, %value = tvm_ffi.call @callee(%input) : (!tvm_ffi.bool) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return
}

// -----

func.func @missing() {
  %result = tvm_ffi.monad.call @absent() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

tvm_ffi.func @callee(!tvm_ffi.int) -> !tvm_ffi.c.tensor
func.func @wrong_arguments(%input: !tvm_ffi.bool) {
  %result = tvm_ffi.monad.call @callee(%input) : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return
}

// -----

tvm_ffi.func @callee() -> !tvm_ffi.c.tensor
func.func @wrong_payload() {
  // expected-error@+1 {{Result payload must match the callee result}}
  %result = tvm_ffi.monad.call @callee() : () -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return
}

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @empty_payload() -> !tvm_ffi.monad.result<>

// -----

func.func @failure_with_payload(%value: i64) {
  // expected-error@+1 {{requires zero operands}}
  %result = "tvm_ffi.monad.fail"(%value) : (i64) -> !tvm_ffi.monad.result<!tvm_ffi.int>

  return
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_integer(!tvm_ffi.monad.result<i64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func private @invalid_object(!tvm_ffi.monad.result<!tvm_ffi.object>)

// -----

func.func @missing_payload(%ready: i1) {
  // expected-error@+1 {{payload count must match Result count}}
  %value = tvm_ffi.monad.pure : () -> !tvm_ffi.monad.result<!tvm_ffi.int>

  return
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

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ResultType parameter}}
func.func @nested(%value: !tvm_ffi.monad.result<!tvm_ffi.monad.result<!tvm_ffi.int>>) {

  return
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

func.func @wrong_pure_payload(%value: i32) {
  // expected-error@+1 {{payload type must match Result type}}
  %result = tvm_ffi.monad.pure %value : (i32) -> !tvm_ffi.monad.result<!tvm_ffi.int>

  return
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

func.func @zero_fail_results() {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.fail"() : () -> ()

  return
}

// -----

func.func @zero_pure_results() {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi.monad.pure"() : () -> ()

  return
}

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @empty()

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @monadic() -> !tvm_ffi.monad.result<!tvm_ffi.int>

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi.func @native() -> i64

// -----

tvm_ffi.func @wrong_flag(%flag: i1, %payload: !tvm_ffi.any) -> !tvm_ffi.int {
  // expected-error@+1 {{operand #0 must be}}
  "tvm_ffi.return"(%flag, %payload) : (i1, !tvm_ffi.any) -> ()
}

// -----

tvm_ffi.func @wrong_payload(%flag: !tvm_ffi.bool, %payload: !tvm_ffi.int) -> !tvm_ffi.int {
  // expected-error@+1 {{operand #1 must be}}
  "tvm_ffi.return"(%flag, %payload) : (!tvm_ffi.bool, !tvm_ffi.int) -> ()
}

// -----

tvm_ffi.func @missing_flag(%payload: !tvm_ffi.any) -> !tvm_ffi.int {
  // expected-error@+1 {{expected 2 operands}}
  "tvm_ffi.return"(%payload) : (!tvm_ffi.any) -> ()
}

// -----

func.func @not_result(%input: i64) {
  // expected-error@+1 {{operand #0 must be}}
  %success, %value = tvm_ffi.monad.unpack %input : (i64) -> (i1, i64)
  return
}

// -----

func.func @wrong_payload(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  // expected-error@+2 {{inferred type}}
  // expected-error@+1 {{failed to infer returned types}}
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> (i1, !tvm_ffi.int)
  return
}
