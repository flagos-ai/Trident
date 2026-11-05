// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s
// RUN: trident-core-opt %s --canonicalize --lower-tvm-ffi-monad --canonicalize --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each -o /dev/null

// Flattening must not discard bind or yield attributes.
// CHECK-LABEL: func.func @attributes(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: tvm_ffi.monad.bind : () -> () {
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK: } {test.keep}
// CHECK: tvm_ffi.monad.bind : () -> () {
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK: tvm_ffi.monad.yield {test.keep}
func.func @attributes(%x: !tvm_ffi.int) {
  %pure = tvm_ffi.monad.pure %x : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.monad.bind %pure : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%a: !tvm_ffi.int):
    func.call @record(%a, %a) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield
  } {test.keep}
  tvm_ffi.monad.bind %pure : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%a: !tvm_ffi.int):
    func.call @record(%a, %a) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield {test.keep}
  }
  return
}

// CHECK-LABEL: func.func @empty_inputs(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: call @record(%[[X]], %[[X]])
// CHECK-NEXT: return
func.func @empty_inputs(%x: !tvm_ffi.int) {
  tvm_ffi.monad.bind : () -> () {
    func.call @record(%x, %x) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield
  }
  return
}

// Remove pure inputs on both sides of an unknown input without unguarding effects.
// CHECK-LABEL: func.func @mixed_inputs(
// CHECK-SAME: %[[R:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.bind %[[R]] :
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK-NEXT: call @record(%[[X]], %[[B]])
// CHECK-NEXT: call @record(%[[B]], %[[X]])
// CHECK: return %[[OUTPUT]]
func.func @mixed_inputs(%r: !tvm_ffi.monad.result<!tvm_ffi.int>, %x: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %pure = tvm_ffi.monad.pure %x : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  %output = tvm_ffi.monad.bind %pure, %r, %pure : (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%a: !tvm_ffi.int, %b: !tvm_ffi.int, %c: !tvm_ffi.int):
    func.call @record(%a, %b) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    func.call @record(%b, %c) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    %result = tvm_ffi.monad.pure %b : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// Pure-only, effect-only continuations inline once at their original position.
// CHECK-LABEL: func.func @pure_effects(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[Y:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: call @record(%[[X]], %[[Y]])
// CHECK-NEXT: call @record(%[[Y]], %[[X]])
// CHECK-NEXT: return
func.func @pure_effects(%x: !tvm_ffi.int, %y: !tvm_ffi.int) {
  %first = tvm_ffi.monad.pure %x : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  %second = tvm_ffi.monad.pure %y : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.monad.bind %first, %second : (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%a: !tvm_ffi.int, %b: !tvm_ffi.int):
    func.call @record(%a, %b) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    func.call @record(%b, %a) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield
  }
  return
}

// Result success is independent of a false Bool payload. Independent pure inputs
// and multiple output forwarding must preserve the tensor reference.
// CHECK-LABEL: func.func @pure_payloads(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK-NEXT: %[[FALSE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool false
// CHECK-NEXT: call @record_bool(%[[FALSE]])
// CHECK-NEXT: %[[TENSOR_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[TENSOR]]
// CHECK-NEXT: %[[BOOL_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[FALSE]]
// CHECK-NEXT: return %[[TENSOR_RESULT]], %[[BOOL_RESULT]]
func.func @pure_payloads(%tensor: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.bool>) {
  %false = tvm_ffi.constant.bool false
  %input_0 = tvm_ffi.monad.pure %false : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
  %input_1 = tvm_ffi.monad.pure %tensor : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  %output:2 = tvm_ffi.monad.bind %input_1, %input_0 : (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.bool>) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.bool>) {
  ^bb0(%x: !tvm_ffi.tensor, %flag: !tvm_ffi.bool):
    func.call @record_bool(%flag) : (!tvm_ffi.bool) -> ()
    %result_0 = tvm_ffi.monad.pure %x : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    %result_1 = tvm_ffi.monad.pure %flag : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
    tvm_ffi.monad.yield %result_0, %result_1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.bool>
  }
  return %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.bool>
}

func.func private @record(!tvm_ffi.int, !tvm_ffi.int)
func.func private @record_bool(!tvm_ffi.bool)
