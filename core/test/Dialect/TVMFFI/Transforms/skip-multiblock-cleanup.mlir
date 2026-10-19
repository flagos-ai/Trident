// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup > %t.once
// RUN: trident-core-opt %t.once --materialize-tvm-ffi-cleanup --verify-each | diff %t.once -

// An empty function region requires no processing.
// CHECK-LABEL: func.func private @external()
func.func private @external() -> !tvm_ffi.tensor

// Direct operations in the multi-block function remain unchanged. Each visited
// nested single-block region independently acquires exports and releases locals.
// CHECK-LABEL: func.func @multiple_blocks(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1
// CHECK-NEXT: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup}
// CHECK-NEXT: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]] -> (!tvm_ffi.exception) {
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INNER]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK-NEXT: scf.yield %[[INNER]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[LOCAL]]
// CHECK-NEXT: scf.yield %[[LOCAL]]
// CHECK-NEXT: } {tvm_ffi.cleanup}
// CHECK-NEXT: cf.br ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return {tvm_ffi.retain} %[[OUTPUT]] : !tvm_ffi.exception
func.func @multiple_blocks(%predicate: i1) -> !tvm_ffi.exception {
  %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %output = scf.if %predicate -> !tvm_ffi.exception {
    %inner = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %inner : !tvm_ffi.exception
  } else {
    scf.yield {tvm_ffi.retain} %local : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  cf.br ^exit
^exit:
  return {tvm_ffi.retain} %output : !tvm_ffi.exception
}

// Pure retains do not require a single-block region. Local cleanup and export
// markers remain pending, and a second pass must not retain the payload again.
// CHECK-LABEL: func.func @pure_in_multiple_blocks(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[INPUT]] {tvm_ffi.cleanup}
// CHECK-NEXT: cf.br ^[[PURE_EXIT:bb[0-9]+]](%[[RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.tensor>)
// CHECK-NEXT: ^[[PURE_EXIT]](%[[OUTPUT_RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.tensor>):
// CHECK-NEXT: return {tvm_ffi.retain} %[[OUTPUT_RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.tensor>
func.func @pure_in_multiple_blocks(%input: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %result = tvm_ffi.monad.pure %input {tvm_ffi.cleanup, tvm_ffi.retain} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  cf.br ^exit(%result : !tvm_ffi.monad.result<!tvm_ffi.tensor>)
^exit(%output: !tvm_ffi.monad.result<!tvm_ffi.tensor>):
  return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
