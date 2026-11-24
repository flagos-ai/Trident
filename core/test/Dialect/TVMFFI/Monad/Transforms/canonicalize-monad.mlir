// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s
// RUN: trident-core-opt %s --canonicalize --lower-tvm-ffi-monad --canonicalize --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --verify-each -o /dev/null

// Flattening must not discard bind or yield attributes.
// CHECK-LABEL: func.func @attributes(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: tvm_ffi_monad.bind : () -> () {
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK: } {test.keep}
// CHECK: tvm_ffi_monad.bind : () -> () {
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK: tvm_ffi_monad.yield {test.keep}
func.func @attributes(%x: !tvm_ffi_abi.int) {
  %pure = tvm_ffi_monad.pure %x : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  tvm_ffi_monad.bind %pure : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> () {
  ^bb0(%a: !tvm_ffi_abi.int):
    func.call @record(%a, %a) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    tvm_ffi_monad.yield
  } {test.keep}
  tvm_ffi_monad.bind %pure : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> () {
  ^bb0(%a: !tvm_ffi_abi.int):
    func.call @record(%a, %a) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    tvm_ffi_monad.yield {test.keep}
  }
  return
}

// CHECK-LABEL: func.func @empty_inputs(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK-NEXT: return
func.func @empty_inputs(%x: !tvm_ffi_abi.int) {
  tvm_ffi_monad.bind : () -> () {
    func.call @record(%x, %x) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    tvm_ffi_monad.yield
  }
  return
}

// Remove pure inputs on both sides of an unknown input without unguarding effects.
// CHECK-LABEL: func.func @mixed_inputs(
// CHECK-SAME: %[[R:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi_monad.bind %[[R]] :
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK-NEXT: call @record(%[[X]], %[[B]])
// CHECK-NEXT: call @record(%[[B]], %[[X]])
// CHECK: return %[[OUTPUT]]
func.func @mixed_inputs(%r: !tvm_ffi_monad.result<!tvm_ffi_abi.int>, %x: !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  %pure = tvm_ffi_monad.pure %x : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  %output = tvm_ffi_monad.bind %pure, %r, %pure : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int> {
  ^bb0(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int, %c: !tvm_ffi_abi.int):
    func.call @record(%a, %b) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    func.call @record(%b, %c) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    %result = tvm_ffi_monad.pure %b : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    tvm_ffi_monad.yield %result : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  }
  return %output : !tvm_ffi_monad.result<!tvm_ffi_abi.int>
}

// Pure-only, effect-only continuations inline once at their original position.
// CHECK-LABEL: func.func @pure_effects(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[Y:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: call @record(%[[X]], %[[Y]])
// CHECK-NEXT: call @record(%[[Y]], %[[X]])
// CHECK-NEXT: return
func.func @pure_effects(%x: !tvm_ffi_abi.int, %y: !tvm_ffi_abi.int) {
  %first = tvm_ffi_monad.pure %x : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  %second = tvm_ffi_monad.pure %y : (!tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  tvm_ffi_monad.bind %first, %second : (!tvm_ffi_monad.result<!tvm_ffi_abi.int>, !tvm_ffi_monad.result<!tvm_ffi_abi.int>) -> () {
  ^bb0(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int):
    func.call @record(%a, %b) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    func.call @record(%b, %a) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> ()
    tvm_ffi_monad.yield
  }
  return
}

// Result success is independent of a false Bool payload. Independent pure inputs
// and multiple output forwarding must preserve the tensor reference.
// CHECK-LABEL: func.func @pure_payloads(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_api.tensor)
// CHECK-NEXT: %[[FALSE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.bool false
// CHECK-NEXT: call @record_bool(%[[FALSE]])
// CHECK-NEXT: %[[TENSOR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_monad.pure %[[TENSOR]]
// CHECK-NEXT: %[[BOOL_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_monad.pure %[[FALSE]]
// CHECK-NEXT: return %[[TENSOR_RESULT]], %[[BOOL_RESULT]]
func.func @pure_payloads(%tensor: !tvm_ffi_api.tensor) -> (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.bool>) {
  %false = tvm_ffi_abi.constant.bool false
  %input_0 = tvm_ffi_monad.pure %false : (!tvm_ffi_abi.bool) -> !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
  %input_1 = tvm_ffi_monad.pure %tensor : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  %output:2 = tvm_ffi_monad.bind %input_1, %input_0 : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.bool>) -> (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.bool>) {
  ^bb0(%x: !tvm_ffi_api.tensor, %flag: !tvm_ffi_abi.bool):
    func.call @record_bool(%flag) : (!tvm_ffi_abi.bool) -> ()
    %result_0 = tvm_ffi_monad.pure %x : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
    %result_1 = tvm_ffi_monad.pure %flag : (!tvm_ffi_abi.bool) -> !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
    tvm_ffi_monad.yield %result_0, %result_1 : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
  }
  return %output#0, %output#1 : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_abi.bool>
}

func.func private @record(!tvm_ffi_abi.int, !tvm_ffi_abi.int)
func.func private @record_bool(!tvm_ffi_abi.bool)
