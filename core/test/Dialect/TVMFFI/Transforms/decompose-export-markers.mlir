// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s
// RUN: trident-core-opt %s --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --verify-each > %t
// RUN: trident-core-opt %t --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// Converted function returns and SCF yields preserve pending markers.
// CHECK-LABEL: func.func @container_branch(
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]]
// CHECK-NEXT: scf.yield {tvm_ffi.retain} %[[INPUT]]
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield {tvm_ffi.retain} %[[INPUT]]
// CHECK: return {tvm_ffi.retain} %[[OUTPUT]]
func.func @container_branch(%predicate: i1, %input: !tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
  %output = scf.if %predicate -> !tvm_ffi.tuple<!tvm_ffi.tensor> {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  } else {
    scf.yield {tvm_ffi.retain} %input : !tvm_ffi.tuple<!tvm_ffi.tensor>
  }
  return {tvm_ffi.retain} %output : !tvm_ffi.tuple<!tvm_ffi.tensor>
}

// Bind success yields preserve intent; generated failure yields remain unmarked.
// CHECK-LABEL: func.func @continuation(
// CHECK: scf.if
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure
// CHECK-NEXT: scf.yield {tvm_ffi.retain} %[[PURE]]
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK-NEXT: scf.yield %[[FAILED]]
func.func @continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%payload: !tvm_ffi.tensor):
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  }
  return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
