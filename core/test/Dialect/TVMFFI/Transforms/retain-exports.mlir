// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --lower-tvm-ffi-monad --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// CHECK-LABEL:   func.func @borrowed(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: i64) -> (!tvm_ffi.tensor, !tvm_ffi.tensor, i64) {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.tensor
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return %[[ARG0]], %[[ARG0]], %[[ARG1]] : !tvm_ffi.tensor, !tvm_ffi.tensor, i64
// CHECK-LABEL:   func.func @container_branch(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.c.array) {
// CHECK:             tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @container_result(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           return %[[ARG0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @containers(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> (!tvm_ffi.c.array, !tvm_ffi.c.array, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG2]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
// CHECK:           return %[[ARG0]], %[[ARG1]], %[[ARG2]] : !tvm_ffi.c.array, !tvm_ffi.c.array, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
// CHECK-LABEL:   func.func @conditional(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1) -> !tvm_ffi.exception {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.exception) {
// CHECK:             %[[EXCEPTION_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_1]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           } else {
// CHECK:             %[[EXCEPTION_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectIncRef %[[EXCEPTION_2]] : !tvm_ffi.exception
// CHECK:             tvm_ffi.ObjectDecRef %[[EXCEPTION_2]] : !tvm_ffi.exception
// CHECK:             scf.yield %[[EXCEPTION_2]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectIncRef %[[IF_0]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return %[[IF_0]] : !tvm_ffi.exception
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
// CHECK:             tvm_ffi.ObjectIncRef %[[DOWNCAST_0]] : !tvm_ffi.tensor
// CHECK:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[VAL_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_4]](%[[CONSTANT_3]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_4]] : !tvm_ffi.object
// CHECK:             tvm_ffi.ObjectDecRef %[[UPCAST_0]] : !tvm_ffi.any
// CHECK:             tvm_ffi.ObjectIncRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_5]] : !tvm_ffi.c.array
// CHECK:           } else {
// CHECK:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[VAL_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:             %[[VAL_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_6]](%[[CONSTANT_5]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_6]] : !tvm_ffi.object
// CHECK:             tvm_ffi.ObjectIncRef %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:             tvm_ffi.ObjectDecRef %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:             scf.yield %[[VAL_7]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectIncRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[IF_0]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_3]] : !tvm_ffi.any
// CHECK:           return %[[IF_0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @failed() -> !tvm_ffi.c.array {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:           %[[VAL_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// CHECK:           %[[VAL_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.FunctionCall %[[VAL_0]](%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_0]] : !tvm_ffi.object
// CHECK:           tvm_ffi.ObjectIncRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           tvm_ffi.ObjectDecRef %[[VAL_1]] : !tvm_ffi.c.array
// CHECK:           return %[[VAL_1]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @native_result(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.c.array
// CHECK:           return %[[ARG0]] : !tvm_ffi.c.array
// CHECK-LABEL:   func.func @scalar(
// CHECK-SAME:                      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i64) -> i64 {
// CHECK:           return %[[ARG0]] : i64
// CHECK-LABEL:   func.func @scalar_cleanup(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i64) -> i64 {
// CHECK:           %[[EXCEPTION_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK:           tvm_ffi.ObjectDecRef %[[EXCEPTION_0]] : !tvm_ffi.exception
// CHECK:           return %[[ARG0]] : i64
// CHECK-LABEL:   func.func @scalar_scopes(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i64) -> i64 {
// CHECK:           %[[EXECUTE_REGION_0:[a-zA-Z0-9_$.-]+]] = scf.execute_region -> i64 {
// CHECK:             scf.yield %[[ARG0]] : i64
// CHECK:           cf.br ^[[BLOCK_bb1:bb[0-9]+]](%[[EXECUTE_REGION_0]] : i64)
// CHECK:         ^[[BLOCK_bb1]](%[[VAL_0:[a-zA-Z0-9_$.-]+]]: i64):
// CHECK:           return {tvm_ffi.retain} %[[VAL_0]] : i64
// CHECK-LABEL:   func.func @unit() {
// CHECK:             scf.yield
// CHECK:           return
// CHECK-LABEL:   func.func @unmarked(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK:           tvm_ffi.ObjectIncRef %[[ARG0]] : !tvm_ffi.tensor
// CHECK:           return %[[ARG0]] : !tvm_ffi.tensor
// CHECK-LABEL:   func.func @unmanaged(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK:           return %[[ARG0]] : !tvm_ffi.tensor






// Repeated borrowed operands each get a reference before any local release.
func.func @borrowed(%input: !tvm_ffi.tensor, %number: i64) -> (!tvm_ffi.tensor, !tvm_ffi.tensor, i64) {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain} %input, %input, %number : !tvm_ffi.tensor, !tvm_ffi.tensor, i64
}

// Container type conversion must preserve processed state on branch yields.
func.func @container_branch(%predicate: i1, %input: !tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<!tvm_ffi.tensor>
}

// Semantic Result container references are deferred until decomposition.
func.func @container_result(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>>
}

// Only the Union is reference-classified before semantic container erasure.
func.func @containers(%array: !tvm_ffi.array<!tvm_ffi.tensor>, %tuple: !tvm_ffi.tuple<>, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> (!tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tuple<>, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  return {tvm_ffi.retain} %array, %tuple, %union : !tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tuple<>, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}


// Both a captured outer value and a newly owned branch value escape safely.
func.func @conditional(%predicate: i1) -> !tvm_ffi.exception {
  %outer = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %result = scf.if %predicate -> !tvm_ffi.exception {
    %inner = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %outer : !tvm_ffi.exception
  } else {
    %other = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %other : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %result : !tvm_ffi.exception
}

// A borrow in the bind body becomes a Result reference through the original
// explicit payload retain. Only the terminator export is deferred.
func.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%borrowed: !tvm_ffi.tensor):
    tvm_ffi.ObjectIncRef %borrowed : !tvm_ffi.tensor
    %pure = tvm_ffi.monad.pure %borrowed {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %result : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// Even a statically failed Result is inspected only under its success bit.
func.func @failed() -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>
  return {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// Successful native handle Results retain their unboxed payload only on success.
func.func @native_result(%input: !tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// Scalar and empty exports need no reference adjustments.
func.func @scalar(%number: i64) -> i64 {
  return {tvm_ffi.retain} %number : i64
}

// Scalar exits still release local references.
func.func @scalar_cleanup(%number: i64) -> i64 {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain} %number : i64
}

// Scalar export markers in skipped CFG regions remain pending.
func.func @scalar_scopes(%input: i64) -> i64 {
  %value = scf.execute_region -> i64 {
    scf.yield {tvm_ffi.retain} %input : i64
  }
  cf.br ^exit(%value : i64)
^exit(%result: i64):
  return {tvm_ffi.retain} %result : i64
}

func.func @unit() {
  scf.execute_region {
    scf.yield {tvm_ffi.retain}
  }
  return {tvm_ffi.retain}
}

// Marked borrowed exports retain even without local cleanup credits.
func.func @unmarked(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  return {tvm_ffi.retain} %input : !tvm_ffi.tensor
}

// Unmarked exits request no automatic references.
func.func @unmanaged(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  return %input : !tvm_ffi.tensor
}
