// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --canonicalize --verify-each | FileCheck %s
// RUN: trident-core-opt %s --decompose-tvm-ffi --convert-scf-to-cf --canonicalize --verify-each | FileCheck %s --check-prefix=CFG
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM


// CHECK-LABEL: tvm_ffi.func @checked_continuation(
// CHECK: %[[LOOKUP:[a-zA-Z0-9_]+]]:2 = func.call @lookup()
// CHECK: %[[INITIAL:[a-zA-Z0-9_]+]] = scf.if %[[LOOKUP]]#1
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]], %[[CALLEE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INITIAL]]
// CHECK: scf.if %[[LOOKUP_OK]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR:[a-zA-Z0-9_]+]])
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = scf.if %[[STATUS]]#1
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[FIRST]]
// CHECK: scf.if %[[OK]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// CHECK: } else {
// CHECK-NOT: call @checked_call
// CHECK: tvm_ffi.monad.fail
// CHECK: tvm_ffi.return
// LLVM-LABEL: llvm.func @checked_continuation(
// LLVM: llvm.call @checked_call
// LLVM: llvm.cond_br
// LLVM: llvm.call @checked_call
// LLVM: llvm.cond_br
// LLVM: llvm.store
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_checked_continuation(
// LLVM: llvm.return
// CHECK-LABEL: tvm_ffi.func @none_continuation(
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: scf.if %[[OK]]
// CHECK: tvm_ffi.tensor.copy_
// CHECK: tvm_ffi.monad.pure
// CHECK: } else {
// CHECK-NOT: tvm_ffi.tensor.copy_
// CHECK: tvm_ffi.monad.fail
// CHECK: tvm_ffi.return
// CFG-LABEL: tvm_ffi.func @none_continuation(
// CFG: ^[[MERGE:bb[0-9]+]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>):
// CFG: %[[OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[RESULT]]
// CFG: cf.cond_br %[[OK]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CFG: ^[[SUCCESS]]:
// CFG: tvm_ffi.tensor.copy_
// CFG: cf.br ^[[EXIT:bb[0-9]+]]
// CFG: ^[[FAILURE]]:
// CFG-NOT: tvm_ffi.tensor.copy_
// CFG: tvm_ffi.monad.fail
// CFG: cf.br ^[[EXIT]]
// CFG: ^[[EXIT]](%[[RETURN:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>):
// CFG: tvm_ffi.return {tvm_ffi.retain} %[[RETURN]]
// LLVM-LABEL: llvm.func @none_continuation(
// LLVM: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.return

// Calls transfer their owned payloads into Results. Explicit cleanup releases
// each Result after the receiving scope acquires its own export reference.
tvm_ffi.func @checked_continuation(%tensor: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {emit_tvm_ffi_abi} {
  %handle, %lookup = func.call @lookup() : () -> (!tvm_ffi.function, i1)
  %ready = scf.if %lookup -> (!tvm_ffi.monad.result<!tvm_ffi.function>) {
    %pure = tvm_ffi.monad.pure %handle {tvm_ffi.cleanup} : (!tvm_ffi.function) -> !tvm_ffi.monad.result<!tvm_ffi.function>
    scf.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.function>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>
    scf.yield {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.function>
  } {tvm_ffi.cleanup}
  %output = tvm_ffi.monad.bind %ready : (!tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%callee: !tvm_ffi.function):
    %value, %status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.function, !tvm_ffi.tensor) -> (!tvm_ffi.tensor, i1)
    %first = scf.if %status -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>) {
      %pure = tvm_ffi.monad.pure %value {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
      scf.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    } else {
      %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      scf.yield {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    } {tvm_ffi.cleanup}
    %continued = tvm_ffi.monad.bind %first : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
    ^bb0(%unused: !tvm_ffi.tensor):
      // Deliberately independent of the first payload: effects still short-circuit.
      %second, %second_status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.function, !tvm_ffi.tensor) -> (!tvm_ffi.tensor, i1)
      %last = scf.if %second_status -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>) {
        %pure = tvm_ffi.monad.pure %second {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
        scf.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      } else {
        %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
        scf.yield {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      } {tvm_ffi.cleanup}
      tvm_ffi.monad.yield {tvm_ffi.retain} %last : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    } {tvm_ffi.cleanup}
    tvm_ffi.monad.yield {tvm_ffi.retain} %continued : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  } {tvm_ffi.cleanup}
  tvm_ffi.return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// A None payload preserves the status dependency for independent side effects.
tvm_ffi.func @none_continuation(%ready: i1, %source: !tvm_ffi.tensor, %destination: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %none = tvm_ffi.constant.none
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
    %pure = tvm_ffi.monad.pure %none {tvm_ffi.cleanup} : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield {tvm_ffi.retain} %fail : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%payload: !tvm_ffi.none):
    tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.tensor
    %pure = tvm_ffi.monad.pure %payload {tvm_ffi.cleanup} : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } {tvm_ffi.cleanup}
  tvm_ffi.return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}

func.func private @lookup() -> (!tvm_ffi.function, i1)
func.func private @checked_call(!tvm_ffi.function, !tvm_ffi.tensor) -> (!tvm_ffi.tensor, i1)
