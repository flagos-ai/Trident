// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-monad --canonicalize --verify-each | FileCheck %s
// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-monad --convert-scf-to-cf --canonicalize --verify-each | FileCheck %s --check-prefix=CFG
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: tvm_ffi.func @checked_continuation(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]], %[[LOOKUP:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.clone"
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]]:2 = scf.if %[[LOOKUP]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]], %[[STATUS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[TENSOR]])
// CHECK: %[[INNER:[a-zA-Z0-9_]+]]:2 = scf.if %[[STATUS]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]], %[[SECOND_STATUS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[TENSOR]])
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = scf.if %[[SECOND_STATUS]]
// CHECK: scf.yield %[[SECOND]]
// CHECK: } else {
// CHECK: scf.yield %[[EMPTY:[a-zA-Z0-9_]+]]
// CHECK: scf.yield %[[PAYLOAD]], %[[SECOND_STATUS]]
// CHECK: } else {
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: tvm_ffi.return
// LLVM-LABEL: llvm.func @checked_continuation(
// LLVM: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.cond_br
// LLVM: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.cond_br
// LLVM: llvm.store
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_checked_continuation(
// LLVM: llvm.cond_br
// LLVM: llvm.store
// LLVM: llvm.return

tvm_ffi.func @checked_continuation(%tensor: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {emit_tvm_ffi_abi} {
  %handle, %lookup = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function, i1
  %ready = scf.if %lookup -> (!tvm_ffi.monad.result<!tvm_ffi.function>) {
    %pure = tvm_ffi.monad.pure %handle : (!tvm_ffi.function) -> !tvm_ffi.monad.result<!tvm_ffi.function>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.function>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>
    scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.function>
  }
  %output = tvm_ffi.monad.bind %ready : (!tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  ^bb0(%callee: !tvm_ffi.function):
    %value, %status = tvm_ffi.FunctionCall %callee(%tensor) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor, i1
    %first = scf.if %status -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>) {
      %pure = tvm_ffi.monad.pure %value : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
      scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    } else {
      %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    }
    %continued = tvm_ffi.monad.bind %first : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
    ^bb0(%unused: !tvm_ffi.tensor):
      // Deliberately independent of the first payload: effects still short-circuit.
      %second, %second_status = tvm_ffi.FunctionCall %callee(%tensor) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor, i1
      %last = scf.if %second_status -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>) {
        %pure = tvm_ffi.monad.pure %second : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
        scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      } else {
        %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.tensor>
        scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.tensor>
      }
      tvm_ffi.monad.yield %last : !tvm_ffi.monad.result<!tvm_ffi.tensor>
    }
    tvm_ffi.monad.yield %continued : !tvm_ffi.monad.result<!tvm_ffi.tensor>
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// A None payload preserves the status dependency for independent side effects.
// CHECK-LABEL: tvm_ffi.func @none_continuation(
// CHECK-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1, %[[SOURCE:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[DESTINATION:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi.none) {
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: tvm_ffi.tensor.copy_ %[[DESTINATION]], %[[SOURCE]]
// CHECK: scf.yield %[[PAYLOAD]] : !tvm_ffi.none
// CHECK: } else {
// CHECK-NOT: tvm_ffi.tensor.copy_
// CHECK: scf.yield %[[FAILED_NONE:[a-zA-Z0-9_]+]] : !tvm_ffi.none
// CHECK: %[[RETURN:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULT]], %[[READY]] : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: tvm_ffi.return %[[RETURN]]
// LLVM-LABEL: llvm.func @none_continuation(
// LLVM: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.return
// CFG-LABEL: tvm_ffi.func @none_continuation(
// CFG-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1, %[[SOURCE:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[DESTINATION:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CFG: ^[[MERGE:bb[0-9]+]](%[[PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi.none, %[[STATUS:[a-zA-Z0-9_]+]]: i1):
// CFG: ^[[UNPACKED:bb[0-9]+]](%[[ANY:[a-zA-Z0-9_]+]]: !tvm_ffi.any):
// CFG: cf.cond_br %[[STATUS]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CFG: ^[[SUCCESS]]:
// CFG: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]]
// CFG: tvm_ffi.tensor.copy_ %[[DESTINATION]], %[[SOURCE]]
// CFG: cf.br ^[[EXIT:bb[0-9]+]](%[[NONE]], %[[TRUE:[a-zA-Z0-9_]+]]
// CFG: ^[[FAILURE]]:
// CFG-NOT: tvm_ffi.tensor.copy_
// CFG: cf.br ^[[EXIT]](%[[FAILED:[a-zA-Z0-9_]+]], %[[FALSE:[a-zA-Z0-9_]+]]
// CFG: ^[[EXIT]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.none, %[[OK:[a-zA-Z0-9_]+]]: i1):
// CFG: %[[RETURN:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULT]], %[[OK]]
// CFG: tvm_ffi.return %[[RETURN]]
tvm_ffi.func @none_continuation(%ready: i1, %source: !tvm_ffi.tensor, %destination: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %none = tvm_ffi.constant.none
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
    %pure = tvm_ffi.monad.pure %none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %fail : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%payload: !tvm_ffi.none):
    tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.tensor
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<!tvm_ffi.none>
}
