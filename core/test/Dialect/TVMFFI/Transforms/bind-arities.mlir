// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --verify-each | FileCheck %s --check-prefix=IR
// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --check-prefix=LOWER --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// IR-LABEL: func.func @empty(
// IR: tvm_ffi.monad.bind : () -> () {
// IR-NEXT: call @record()
// IR-NEXT: tvm_ffi.monad.yield
// LOWER-LABEL: func.func @empty(
// LOWER: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// LOWER-NEXT: scf.if %[[TRUE]] {
// LOWER-NEXT: call @record()
// LOWER-NEXT: }
// LOWER-NEXT: return
// LLVM-LABEL: llvm.func @empty(
// LLVM: llvm.call @record()
// LLVM: llvm.return
func.func @empty() {
  tvm_ffi.monad.bind : () -> () {
    func.call @record() : () -> ()
    tvm_ffi.monad.yield {tvm_ffi.retain}
  }
  return {tvm_ffi.retain}
}

// No inputs means the continuation executes unconditionally and may produce
// any number of Results.
// IR-LABEL: func.func @no_inputs(
// IR: tvm_ffi.monad.bind : () -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
// LOWER-LABEL: func.func @no_inputs(
// LOWER-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// LOWER: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// LOWER-NEXT: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = scf.if %[[TRUE]]
// LOWER-NEXT: %[[PURE:[a-zA-Z0-9_]+]]:2 = tvm_ffi.monad.pure %[[VALUE]], %[[VALUE]]
// LOWER-NEXT: scf.yield {tvm_ffi.retain} %[[PURE]]#0, %[[PURE]]#1
// LOWER: return {tvm_ffi.retain} %[[OUTPUT]]#0, %[[OUTPUT]]#1
// LLVM-LABEL: llvm.func @no_inputs(
// LLVM: llvm.return
func.func @no_inputs(%value: !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
  %output:2 = tvm_ffi.monad.bind : () -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) {
    %pure:2 = tvm_ffi.monad.pure %value, %value : (!tvm_ffi.int, !tvm_ffi.int) -> (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>)
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure#0, %pure#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  return {tvm_ffi.retain} %output#0, %output#1 : !tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>
}

// Every input must succeed before the side effect executes. The no-output
// failure path needs neither a failed Result nor an else branch.
// IR-LABEL: func.func @no_outputs(
// IR: tvm_ffi.monad.bind
// IR-SAME: -> () {
// LOWER-LABEL: func.func @no_outputs(
// LOWER-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>
// LOWER: %[[OK0:[a-zA-Z0-9_]+]], %[[VALUE0:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[FIRST]]
// LOWER: %[[OK1:[a-zA-Z0-9_]+]], %[[VALUE1:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[SECOND]]
// LOWER-NEXT: %[[READY:[a-zA-Z0-9_]+]] = arith.andi %[[OK0]], %[[OK1]]
// LOWER-NEXT: scf.if %[[READY]] {
// LOWER-NEXT: %[[ARG0:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[VALUE0]] : !tvm_ffi.any -> !tvm_ffi.int
// LOWER-NEXT: %[[ARG1:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[VALUE1]] : !tvm_ffi.any -> !tvm_ffi.int
// LOWER-NEXT: call @record_values(%[[ARG0]], %[[ARG1]])
// LOWER-NEXT: }
// LOWER-NEXT: return
// LLVM-LABEL: llvm.func @no_outputs(
// LLVM: llvm.cond_br %[[READY:[a-zA-Z0-9_]+]], ^[[SUCCESS:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// LLVM: ^[[SUCCESS]]:
// LLVM: llvm.call @record_values
// LLVM-NEXT: llvm.br ^[[EXIT]]
// LLVM: ^[[EXIT]]:
// LLVM-NEXT: llvm.return
func.func @no_outputs(%first: !tvm_ffi.monad.result<!tvm_ffi.int>, %second: !tvm_ffi.monad.result<!tvm_ffi.int>) {
  tvm_ffi.monad.bind %first, %second : (!tvm_ffi.monad.result<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.int>) -> () {
  ^bb0(%a: !tvm_ffi.int, %b: !tvm_ffi.int):
    func.call @record_values(%a, %b) : (!tvm_ffi.int, !tvm_ffi.int) -> ()
    tvm_ffi.monad.yield {tvm_ffi.retain}
  }
  return {tvm_ffi.retain}
}

func.func private @record()
func.func private @record_values(!tvm_ffi.int, !tvm_ffi.int)
