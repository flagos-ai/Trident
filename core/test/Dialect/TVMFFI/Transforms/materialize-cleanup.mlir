// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s

// Retain the exported alias before releasing local references in reverse order.
// Unmarked references are unmanaged by this pass.
// CHECK-LABEL: func.func @aliases(
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[UNMANAGED:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[FIRST]]
// CHECK: tvm_ffi.ObjectIncRef %[[ALIAS]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-NEXT: return %[[ALIAS]]
func.func @aliases() -> !tvm_ffi.any {
  %first = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %second = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %unmanaged = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %alias = tvm_ffi.upcast %first : !tvm_ffi.exception -> !tvm_ffi.any
  return {tvm_ffi.retain} %alias : !tvm_ffi.any
}

// CHECK-LABEL: func.func @container_results(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>
// CHECK: %[[ELEMENTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[INPUT]]
// CHECK: tvm_ffi.ObjectDecRef %[[ELEMENTS]]#1
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENTS]]#0
// CHECK-NEXT: return
func.func @container_results(%input: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) {
  %elements:2 = tvm_ffi.tuple.unconstruct %input {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) -> (!tvm_ffi.tensor, !tvm_ffi.tensor)
  return {tvm_ffi.retain}
}

// CHECK-LABEL: tvm_ffi.func @continuation(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.tensor>
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.bind %[[INPUT]]
// CHECK: ^bb0(%[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
// CHECK: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[UNIT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LOCAL]]
// CHECK-NEXT: tvm_ffi.monad.yield %[[UNIT]]
// CHECK: tvm_ffi.return %[[OUTPUT]]
tvm_ffi.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%borrowed: !tvm_ffi.tensor):
    %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    %none = tvm_ffi.constant.none
    %unit = tvm_ffi.monad.pure %none {tvm_ffi.cleanup} : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield {tvm_ffi.retain} %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
  } {tvm_ffi.cleanup}
  tvm_ffi.return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// Each Result has its own success condition, including partial failures.
// CHECK-LABEL: func.func @independent_results(
// CHECK: %[[RESULTS:[a-zA-Z0-9_]+]]:2 = call @produce
// CHECK: %[[PAYLOAD1_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD1:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULTS]]#1
// CHECK-NEXT: scf.if %[[PAYLOAD1_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD1]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[PAYLOAD0_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD0:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULTS]]#0
// CHECK-NEXT: scf.if %[[PAYLOAD0_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD0]]
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @independent_results() {
  %results:2 = call @produce() {tvm_ffi.cleanup} : () -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
  return {tvm_ffi.retain}
}
func.func private @produce() -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)

// Native handle Results retain their payload types when unpacked for cleanup.
// CHECK-LABEL: func.func @native_handles(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @produce_handle
// CHECK: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[PAYLOAD]] : !tvm_ffi.function
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]] : !tvm_ffi.object
// CHECK-NEXT: return
func.func @native_handles() {
  %handle = tvm_ffi.FunctionGetGlobal "ffi.Array" {tvm_ffi.cleanup} : !tvm_ffi.object
  %result = call @produce_handle() {tvm_ffi.cleanup} : () -> !tvm_ffi.monad.result<!tvm_ffi.function>
  return {tvm_ffi.retain}
}
func.func private @produce_handle() -> !tvm_ffi.monad.result<!tvm_ffi.function>


// Child scopes borrow captures; cleanup for child locals precedes their yield.
// CHECK-LABEL: func.func @nested(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: scf.if %[[PREDICATE]] {
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: tvm_ffi.ObjectIncRef %[[OUTER]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTER]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK-NEXT: } else {
// CHECK: %[[OTHER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OTHER]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTER]]
// CHECK-NEXT: return
func.func @nested(%predicate: i1) {
  %outer = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  scf.if %predicate {
    %inner = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    tvm_ffi.ObjectIncRef %outer : !tvm_ffi.exception
    tvm_ffi.ObjectDecRef %outer : !tvm_ffi.exception
  } else {
    %other = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  }
  return {tvm_ffi.retain}
}

// Parent results own independent references retained before child cleanup.
// CHECK-LABEL: func.func @region_output(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]]
// CHECK: %[[THEN:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[THEN]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[THEN]]
// CHECK-NEXT: scf.yield %[[THEN]]
// CHECK: } else {
// CHECK: %[[ELSE:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ELSE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELSE]]
// CHECK-NEXT: scf.yield %[[ELSE]]
// CHECK: tvm_ffi.ObjectDecRef %[[OUTPUT]]
// CHECK-NEXT: return
func.func @region_output(%predicate: i1) {
  %output = scf.if %predicate -> !tvm_ffi.exception {
    %then = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %then : !tvm_ffi.exception
  } else {
    %else = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %else : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain}
}

// Semantic container references are deferred until decomposition.
// CHECK-LABEL: func.func @semantic_container(
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK-NEXT: return
func.func @semantic_container() {
  %tuple = tvm_ffi.tuple.construct() {tvm_ffi.cleanup} : () -> !tvm_ffi.tuple<>
  %none = tvm_ffi.constant.none
  return {tvm_ffi.retain}
}
