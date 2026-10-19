// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// Explicit markers cover newly acquired references.
// CHECK-LABEL: func.func @allocations(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK: %[[LITERAL:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.literal
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal
// CHECK: %[[CALLED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[INPUT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CALLED]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LITERAL]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CLONE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[EXCEPTION]]
// CHECK-NEXT: return
func.func @allocations(%input: !tvm_ffi.tensor) {
  %exception = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %clone = tvm_ffi.tensor.clone %input {tvm_ffi.cleanup} : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %literal = tvm_ffi.tensor.literal dense<1.0> : tensor<1xf32> {tvm_ffi.cleanup} : !tvm_ffi.tensor
  %handle = tvm_ffi.FunctionGetGlobal "test.identity" {tvm_ffi.cleanup} : !tvm_ffi.object
  %called = tvm_ffi.FunctionCall %handle(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
  return {tvm_ffi.retain}
}

// Mixed results acquire credits independently; aliases and scalars stay borrowed.
// CHECK-LABEL: func.func @containers(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[NUMBER:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[INPUT]], %[[NUMBER]])
// CHECK: %[[ELEMENTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[TUPLE]]
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.array.create(%[[INPUT]])
// CHECK: %[[INDEX:[a-zA-Z0-9_]+]] = arith.constant 0
// CHECK: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.array.get %[[ARRAY]][%[[INDEX]]]
// CHECK: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[ELEMENT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENTS]]#0
// CHECK-NEXT: return
func.func @containers(%input: !tvm_ffi.tensor) {
  %number = tvm_ffi.constant.int 1
  %tuple = tvm_ffi.tuple.construct(%input, %number) {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>
  %elements:2 = tvm_ffi.tuple.unconstruct %tuple {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>) -> (!tvm_ffi.tensor, !tvm_ffi.int)
  %array = tvm_ffi.array.create(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.array<!tvm_ffi.tensor>
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %array[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.tensor
  %alias = tvm_ffi.upcast %element : !tvm_ffi.tensor -> !tvm_ffi.any
  return {tvm_ffi.retain}
}

// Unmarked references can be transferred; marked references release once.
// CHECK-LABEL: func.func @overrides(
// CHECK: %[[TRANSFERRED:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[TRANSFERRED]]
// CHECK: %[[MULTIPLE:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[MULTIPLE]]
// CHECK-NEXT: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @overrides() {
  %transferred = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %result = tvm_ffi.monad.pure %transferred {tvm_ffi.cleanup} : (!tvm_ffi.exception) -> !tvm_ffi.monad.result<!tvm_ffi.exception>
  %multiple = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain}
}

// Result cleanup is type-dependent; failed Results never read payloads.
// CHECK-LABEL: func.func @results(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK: %[[NUMBER:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[RESULTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[INPUT]], %[[NUMBER]]
// CHECK: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULTS]]#0
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @results(%input: !tvm_ffi.tensor) {
  tvm_ffi.ObjectIncRef %input : !tvm_ffi.tensor
  %number = tvm_ffi.constant.int 1
  %results:2 = tvm_ffi.monad.pure %input, %number {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.int>)
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return {tvm_ffi.retain}
}

// Scalar and zero-result markers are consumed without reference operations.
// CHECK-LABEL: func.func @scalar_and_empty_cleanup()
// CHECK-NEXT: %[[NUMBER:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 7
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NUMBER]] : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
// CHECK-NEXT: tvm_ffi.monad.bind : () -> () {
// CHECK-NEXT: tvm_ffi.monad.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @scalar_and_empty_cleanup() {
  %number = tvm_ffi.constant.int 7 {tvm_ffi.cleanup}
  %result = tvm_ffi.monad.pure %number {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.monad.bind : () -> () {
    tvm_ffi.monad.yield {tvm_ffi.retain}
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain}
}
