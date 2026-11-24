// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @missing() {
  %ok, %value = tvm_ffi_api.call @absent() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return
}

// -----

tvm_ffi_api.func @callee(!tvm_ffi_abi.int) -> !tvm_ffi_abi.tensor
func.func @wrong_arguments(%input: !tvm_ffi_abi.bool) {
  // expected-error@+1 {{operand types must match the callee arguments}}
  %ok, %value = tvm_ffi_api.call @callee(%input) : (!tvm_ffi_abi.bool) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return
}

// -----

tvm_ffi_api.func @callee() -> !tvm_ffi_abi.tensor
func.func @wrong_payload() {
  // expected-error@+1 {{result #1 must be}}
  %ok, %value = tvm_ffi_api.call @callee() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.int)
  return
}

// -----

tvm_ffi_api.func @callee() -> !tvm_ffi_abi.none
func.func @missing_payload() {
  // expected-error@+1 {{expected 2 results}}
  %ok = "tvm_ffi_api.call"() {callee = @callee} : () -> !tvm_ffi_abi.bool
  return
}

// -----

tvm_ffi_api.func @callee() -> !tvm_ffi_abi.none
func.func @wrong_flag() {
  // expected-error@+1 {{result #0 must be}}
  %ok, %value = tvm_ffi_api.call @callee() : () -> (!tvm_ffi_abi.int, !tvm_ffi_api.any)
  return
}

// -----

func.func private @callee(!tvm_ffi_abi.int, !llvm.ptr) -> i32
func.func @wrong_lowered_arguments(%input: !tvm_ffi_abi.bool) {
  // expected-error@+1 {{operand types must match the lowered callee arguments}}
  %ok, %value = tvm_ffi_api.call @callee(%input) : (!tvm_ffi_abi.bool) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return
}

// -----

func.func @missing() {
  %result = tvm_ffi_monad.call @absent() : () -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  return
}

// -----

tvm_ffi_api.func @callee(!tvm_ffi_abi.int) -> !tvm_ffi_abi.tensor
func.func @wrong_arguments(%input: !tvm_ffi_abi.bool) {
  %result = tvm_ffi_monad.call @callee(%input) : (!tvm_ffi_abi.bool) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  return
}

// -----

tvm_ffi_api.func @callee() -> !tvm_ffi_abi.tensor
func.func @wrong_payload() {
  // expected-error@+1 {{Result payload must match the callee result}}
  %result = tvm_ffi_monad.call @callee() : () -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  return
}

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{failed to parse TVMFFIMonad_ResultType parameter}}
func.func private @empty_payload() -> !tvm_ffi_monad.result<>

// -----

func.func @failure_with_payload(%value: i64) {
  // expected-error@+1 {{requires zero operands}}
  %result = "tvm_ffi_monad.fail"(%value) : (i64) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>

  return
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFIMonad_ResultType parameter}}
func.func private @invalid_integer(!tvm_ffi_monad.result<i64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFIMonad_ResultType parameter}}
func.func private @invalid_object(!tvm_ffi_monad.result<!tvm_ffi_abi.object>)

// -----

func.func @missing_payload(%ready: i1) {
  // expected-error@+1 {{expected SSA operand}}
  %value = tvm_ffi_monad.pure : () -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>

  return
}

// -----

// Missing terminators are a dialect verifier contract, independent of Python.
func.func @missing_yield(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none> {
  // expected-error@+1 {{body must end with tvm_ffi_monad.yield}}
  %output = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none> {
  ^bb_unit(%unit_payload: !tvm_ffi_abi.none):
    %unused = arith.constant 7 : i64
  }
  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.none>
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFIMonad_ResultType parameter}}
func.func @nested(%value: !tvm_ffi_monad.result<!tvm_ffi_monad.result<!tvm_ffi_abi.int>>) {

  return
}

// -----

func.func @wrong_argument(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  // expected-error@+1 {{region arguments must match input Result payloads}}
  %output = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%payload: i32):
    tvm_ffi_monad.yield %input : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }

  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// -----

func.func @wrong_parent() -> !tvm_ffi_monad.result<!tvm_ffi_abi.none> {
  // expected-error@+1 {{must be nested directly in tvm_ffi_monad.bind}}
  tvm_ffi_monad.yield
}

// -----

func.func @wrong_pure_payload(%value: i32) {
  // expected-error@+1 {{payload type must match Result type}}
  %result = tvm_ffi_monad.pure %value : (i32) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>

  return
}

// -----

func.func @wrong_yield(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %other: !tvm_ffi_monad.result<!tvm_ffi_abi.bool>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  %output = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%payload: !tvm_ffi_abi.int):
    // expected-error@+1 {{Result types must match bind outputs}}
    tvm_ffi_monad.yield %other : !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
  }

  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// -----

func.func @zero_fail_results() {
  // expected-error@+1 {{requires at least one Result}}
  "tvm_ffi_monad.fail"() : () -> ()

  return
}

// -----

func.func @zero_pure_results(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{requires one result}}
  "tvm_ffi_monad.pure"(%value) : (!tvm_ffi_abi.int) -> ()

  return
}

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi_api.func @empty()

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi_api.func @monadic() -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>

// -----

// expected-error@+1 {{function must return exactly one C ABI value}}
tvm_ffi_api.func @native() -> i64

// -----

tvm_ffi_api.func @wrong_flag(%flag: i1, %payload: !tvm_ffi_api.any) -> !tvm_ffi_abi.int {
  // expected-error@+1 {{operand #0 must be}}
  "tvm_ffi_api.return"(%flag, %payload) : (i1, !tvm_ffi_api.any) -> ()
}

// -----

tvm_ffi_api.func @wrong_payload(%flag: !tvm_ffi_abi.bool, %payload: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  // expected-error@+1 {{operand #1 must be}}
  "tvm_ffi_api.return"(%flag, %payload) : (!tvm_ffi_abi.bool, !tvm_ffi_abi.int) -> ()
}

// -----

tvm_ffi_api.func @missing_flag(%payload: !tvm_ffi_api.any) -> !tvm_ffi_abi.int {
  // expected-error@+1 {{expected 2 operands}}
  "tvm_ffi_api.return"(%payload) : (!tvm_ffi_api.any) -> ()
}

// -----

func.func @not_result(%input: i64) {
  // expected-error@+1 {{operand #0 must be}}
  %success, %value = tvm_ffi_monad.unpack %input : (i64) -> (i1, i64)
  return
}

// -----

func.func @wrong_payload(%input: !tvm_ffi_monad.result<!tvm_ffi_abi.int>) {
  // expected-error@+2 {{inferred type}}
  // expected-error@+1 {{failed to infer returned types}}
  %success, %value = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> (i1, !tvm_ffi_abi.int)
  return
}

// -----

func.func @multiple_pure_operands(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{requires a single operand}}
  %result = "tvm_ffi_monad.pure"(%value, %value) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  return
}

// -----

func.func @multiple_pure_results(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{requires one result}}
  %results:2 = "tvm_ffi_monad.pure"(%value) : (!tvm_ffi_abi.int) -> (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>)
  return
}
