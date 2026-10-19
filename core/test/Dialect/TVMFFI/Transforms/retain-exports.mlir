// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// Repeated borrowed operands each get a reference before any local release.
// CHECK-LABEL: func.func @borrowed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[NUMBER:[a-zA-Z0-9_]+]]: i64
// CHECK: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LOCAL]]
// CHECK-NEXT: return %[[INPUT]], %[[INPUT]], %[[NUMBER]]
func.func @borrowed(%input: !tvm_ffi.tensor, %number: i64) -> (!tvm_ffi.tensor, !tvm_ffi.tensor, i64) {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain} %input, %input, %number : !tvm_ffi.tensor, !tvm_ffi.tensor, i64
}

// Container type conversion must preserve processed state on branch yields.
// CHECK-LABEL: func.func @container_branch(
// CHECK-NOT: tvm_ffi.ObjectIncRef
func.func @container_branch(%predicate: i1, %input: !tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<!tvm_ffi.tensor>
}

// Semantic Result container references are deferred until decomposition.
// CHECK-LABEL: func.func @container_result(
// CHECK-NOT: tvm_ffi.ObjectIncRef
func.func @container_result(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor>>
}

// Only the Union is reference-classified before semantic container erasure.
// CHECK-LABEL: func.func @containers(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array<!tvm_ffi.tensor>, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<>, %[[UNION:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[UNION]]
// CHECK-NEXT: return %[[ARRAY]], %[[TUPLE]], %[[UNION]]
func.func @containers(%array: !tvm_ffi.array<!tvm_ffi.tensor>, %tuple: !tvm_ffi.tuple<>, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> (!tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tuple<>, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  return {tvm_ffi.retain} %array, %tuple, %union : !tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tuple<>, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}


// Both a captured outer value and a newly owned branch value escape safely.
// CHECK-LABEL: func.func @conditional(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]]
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OUTER]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK-NEXT: scf.yield %[[OUTER]]
// CHECK: } else {
// CHECK: %[[OTHER:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OTHER]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OTHER]]
// CHECK-NEXT: scf.yield %[[OTHER]]
// CHECK: tvm_ffi.ObjectIncRef %[[RESULT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTER]]
// CHECK-NEXT: return %[[RESULT]]
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
// CHECK-LABEL: tvm_ffi.func @continuation(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.tensor>
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.bind %[[INPUT]]
// CHECK: ^bb0(%[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
// CHECK: tvm_ffi.ObjectIncRef %[[BORROWED]]
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[BORROWED]]
// CHECK-NEXT: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[PURE]]
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[RELEASE_PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[RELEASE_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[PURE]]
// CHECK-NEXT: scf.if %[[RELEASE_PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RELEASE_PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.monad.yield %[[PURE]]
// CHECK: %[[RETURN_PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[RETURN_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[RETURN_PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RETURN_PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[CLEANUP_PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[CLEANUP_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CHECK-NEXT: scf.if %[[CLEANUP_PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CLEANUP_PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.return %[[RESULT]]
tvm_ffi.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%borrowed: !tvm_ffi.tensor):
    tvm_ffi.ObjectIncRef %borrowed : !tvm_ffi.tensor
    %pure = tvm_ffi.monad.pure %borrowed {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  } {tvm_ffi.cleanup}
  tvm_ffi.return {tvm_ffi.retain} %result : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// Even a statically failed Result is inspected only under its success bit.
// CHECK-LABEL: func.func @failed(
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK-NEXT: %[[PAYLOAD_OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[FAILED]]
// CHECK-NEXT: scf.if %[[PAYLOAD_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[PAYLOAD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FAILED]]
func.func @failed() -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>
  return {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// Successful native handle Results retain their unboxed payload only on success.
// CHECK-LABEL: func.func @native_result(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.function>
// CHECK-NEXT: %[[HANDLE_OK:[a-zA-Z0-9_]+]], %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK-NEXT: scf.if %[[HANDLE_OK]] {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[HANDLE]] : !tvm_ffi.function
// CHECK-NEXT: }
// CHECK-NEXT: return %[[INPUT]]
func.func @native_result(%input: !tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  return {tvm_ffi.retain} %input : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// Scalar and empty exports need no reference adjustments.
// CHECK-LABEL: func.func @scalar(
// CHECK-SAME: %[[NUMBER:[a-zA-Z0-9_]+]]: i64
// CHECK-NEXT: return %[[NUMBER]]
func.func @scalar(%number: i64) -> i64 {
  return {tvm_ffi.retain} %number : i64
}

// Scalar exits still release local references.
// CHECK-LABEL: func.func @scalar_cleanup(
// CHECK-SAME: %[[NUMBER:[a-zA-Z0-9_]+]]: i64
// CHECK: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LOCAL]]
// CHECK-NEXT: return %[[NUMBER]]
func.func @scalar_cleanup(%number: i64) -> i64 {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain} %number : i64
}

// Scalar export markers in skipped CFG regions remain pending.
// CHECK-LABEL: func.func @scalar_scopes(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: i64
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = scf.execute_region -> i64 {
// CHECK-NEXT: scf.yield %[[INPUT]] : i64
// CHECK-NEXT: }
// CHECK-NEXT: cf.br ^[[EXIT:bb[0-9]+]](%[[VALUE]] : i64)
// CHECK: ^[[EXIT]](%[[RESULT:[a-zA-Z0-9_]+]]: i64):
// CHECK-NEXT: return {tvm_ffi.retain} %[[RESULT]] : i64
func.func @scalar_scopes(%input: i64) -> i64 {
  %value = scf.execute_region -> i64 {
    scf.yield {tvm_ffi.retain} %input : i64
  }
  cf.br ^exit(%value : i64)
^exit(%result: i64):
  return {tvm_ffi.retain} %result : i64
}

// CHECK-LABEL: func.func @unit(
// CHECK-NEXT: scf.execute_region {
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @unit() {
  scf.execute_region {
    scf.yield {tvm_ffi.retain}
  }
  return {tvm_ffi.retain}
}

// Marked borrowed exports retain even without local cleanup credits.
// CHECK-LABEL: func.func @unmarked(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: return %[[INPUT]]
func.func @unmarked(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  return {tvm_ffi.retain} %input : !tvm_ffi.tensor
}

// Unmarked exits request no automatic references.
// CHECK-LABEL: func.func @unmanaged(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK-NEXT: return %[[INPUT]] : !tvm_ffi.tensor
func.func @unmanaged(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  return %input : !tvm_ffi.tensor
}
