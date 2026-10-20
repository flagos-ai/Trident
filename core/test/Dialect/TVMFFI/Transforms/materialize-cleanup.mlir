// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s

// CHECK-LABEL:   func.func @aliases() -> !tvm_ffi.any {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[EXCEPTION_0]] : !tvm_ffi.exception -> !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectIncRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return %[[UPCAST_0]] : !tvm_ffi.any
// CHECK-LABEL:   func.func @container_results(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_0]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.tensor
// CHECK:           return
// CHECK-LABEL:   func.func @continuation(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = arith.constant 0 : i64
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_1]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[ARG0]], %[[TO_0]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.bool
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = arith.constant 1 : i64
// CHECK:           %[[TO_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[CONSTANT_2]] : i64 -> !tvm_ffi.int
// CHECK:           %[[VAL_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.object
// CHECK:           %[[VAL_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_2]](%[[ARG0]], %[[TO_1]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.any
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_2]] : !tvm_ffi.object
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[VAL_1]] : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.c.array) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[VAL_3]] : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK:             %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_3]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[CONSTANT_4]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:             tvm_ffi.ObjectIncRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]](%[[CONSTANT_6]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_6]] : !tvm_ffi.object
// CHECK:             tvm_ffi.ObjectIncRef %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.any
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @independent_results() {
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]]:2 = call @produce() : () -> (!tvm_ffi.c.array, !tvm_ffi.c.array)
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]]#1 : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]]#0 : !tvm_ffi.c.array
// CHECK:           return
// CHECK-LABEL:   func.func @native_handles() {
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = call @produce_handle() : () -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           return
// CHECK-LABEL:   func.func @nested(
// CHECK-SAME:                      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1) {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           scf.if %[[ARG0]] {
// CHECK:             %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:           } else {
// CHECK:             %[[EXCEPTION_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_2]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return
// CHECK-LABEL:   func.func @region_output(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1) {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.exception) {
// CHECK:             %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           } else {
// CHECK:             %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.exception
// CHECK:           return
// CHECK-LABEL:   func.func @semantic_container() {
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]]() : () -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return






// Retain the exported alias before releasing local references in reverse order.
// Unmarked references are unmanaged by this pass.
func.func @aliases() -> !tvm_ffi.any {
  %first = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %second = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %unmanaged = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %alias = tvm_ffi.upcast %first : !tvm_ffi.exception -> !tvm_ffi.any
  return {tvm_ffi.retain} %alias : !tvm_ffi.any
}

func.func @container_results(%input: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) {
  %elements:2 = tvm_ffi.tuple.unconstruct %input {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor>) -> (!tvm_ffi.tensor, !tvm_ffi.tensor)
  return {tvm_ffi.retain}
}

func.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%borrowed: !tvm_ffi.tensor):
    %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    %none = tvm_ffi.constant.none
    %unit = tvm_ffi.monad.pure %none {tvm_ffi.cleanup} : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield {tvm_ffi.retain} %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// Each Result has its own success condition, including partial failures.
func.func @independent_results() {
  %results:2 = call @produce() {tvm_ffi.cleanup} : () -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
  return {tvm_ffi.retain}
}
func.func private @produce() -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)

// Native handle Results retain their payload types when unpacked for cleanup.
func.func @native_handles() {
  %handle = tvm_ffi.FunctionGetGlobal "ffi.Array" {tvm_ffi.cleanup} : !tvm_ffi.object
  %result = call @produce_handle() {tvm_ffi.cleanup} : () -> !tvm_ffi.monad.result<!tvm_ffi.function>
  return {tvm_ffi.retain}
}
func.func private @produce_handle() -> !tvm_ffi.monad.result<!tvm_ffi.function>


// Child scopes borrow captures; cleanup for child locals precedes their yield.
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
func.func @semantic_container() {
  %tuple = tvm_ffi.tuple.construct() {tvm_ffi.cleanup} : () -> !tvm_ffi.tuple<>
  %none = tvm_ffi.constant.none
  return {tvm_ffi.retain}
}
