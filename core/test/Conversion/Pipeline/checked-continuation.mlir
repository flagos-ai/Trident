// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --verify-each | FileCheck %s --implicit-check-not='tvm_ffi_monad.'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi_'

// Result construction is covered by lower-monad.mlir. Here each continuation
// must gate the next call, even when it does not consume the preceding payload.
// CHECK-LABEL: tvm_ffi_api.func @checked_continuation(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[LOOKUP:[a-zA-Z0-9_]+]]:2 = func.call @lookup()
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = scf.if %[[LOOKUP]]#1
// CHECK: %[[READY_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[READY]]
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[READY_PARTS]]#0
// CHECK-NEXT: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if %[[LOOKUP_OK]]
// CHECK-NEXT: %[[CALLEE:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[READY_PARTS]]#1
// CHECK-NEXT: %[[FIRST:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// CHECK-NEXT: %[[FIRST_RESULT:[a-zA-Z0-9_]+]] = scf.if %[[FIRST]]#1
// CHECK: %[[FIRST_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[FIRST_RESULT]]
// CHECK: %[[FIRST_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[FIRST_PARTS]]#0
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = scf.if %[[FIRST_OK]]
// CHECK-NEXT: %[[UNUSED:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[FIRST_PARTS]]#1
// CHECK-NEXT: %[[SECOND:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// CHECK-NEXT: %[[SECOND_RESULT:[a-zA-Z0-9_]+]] = scf.if %[[SECOND]]#1
// CHECK: scf.yield %[[SECOND_RESULT]]
// CHECK: scf.yield %[[INNER]]
// CHECK-NOT: func.call
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[OUTER]]
// CHECK: tvm_ffi_api.return %[[OUTPUT]]#0, %[[OUTPUT]]#1

// A None payload must still gate a payload-independent mutation.
// CHECK-LABEL: tvm_ffi_api.func @none_continuation(
// CHECK-SAME: %[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool, %[[SOURCE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[DEST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[BOXED]]
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: %[[PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[INPUT]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[PARTS]]#0
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK-NEXT: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[PARTS]]#1
// CHECK-NEXT: tvm_ffi_api.tensor.copy_ %[[DEST]], %[[SOURCE]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi_api.tensor.copy_
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = tvm_ffi_api.tuple.unconstruct %[[RESULT]]
// CHECK: tvm_ffi_api.return %[[OUTPUT]]#0, %[[OUTPUT]]#1

// LLVM-LABEL: llvm.func @__tvm_ffi_checked_continuation(
// LLVM: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.call @checked_call(%[[CALLEE:[a-zA-Z0-9_]+]], %[[TENSOR:[a-zA-Z0-9_]+]])
// LLVM: %[[FIRST_OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[FIRST]][1]
// LLVM-NEXT: llvm.cond_br %[[FIRST_OK]]
// LLVM: %[[SECOND:[a-zA-Z0-9_]+]] = llvm.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// LLVM: %[[SECOND_OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[SECOND]][1]
// LLVM-NEXT: llvm.cond_br %[[SECOND_OK]]
// LLVM-LABEL: llvm.func @__tvm_ffi_none_continuation(
// LLVM: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall
// LLVM: llvm.return

tvm_ffi_api.func @checked_continuation(%tensor: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor {
  %handle, %lookup = func.call @lookup() : () -> (!tvm_ffi_abi.function, i1)
  %ready = scf.if %lookup -> (!tvm_ffi_monad.result<!tvm_ffi_abi.function>) {
    %pure = tvm_ffi_monad.pure %handle : (!tvm_ffi_abi.function) -> !tvm_ffi_monad.result<!tvm_ffi_abi.function>
    scf.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.function>
  } else {
    %failed = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.function>
    scf.yield %failed : !tvm_ffi_monad.result<!tvm_ffi_abi.function>
  }
  %output = tvm_ffi_monad.bind %ready : (!tvm_ffi_monad.result<!tvm_ffi_abi.function>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor> {
  ^bb0(%callee: !tvm_ffi_abi.function):
    %value, %status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi_abi.function, !tvm_ffi_abi.tensor) -> (!tvm_ffi_abi.tensor, i1)
    %first = scf.if %status -> (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>) {
      %pure = tvm_ffi_monad.pure %value : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
      scf.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
    } else {
      %failed = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
      scf.yield %failed : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
    }
    %continued = tvm_ffi_monad.bind %first : (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor> {
    ^bb0(%unused: !tvm_ffi_abi.tensor):
      // Deliberately independent of the first payload: effects still short-circuit.
      %second, %second_status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi_abi.function, !tvm_ffi_abi.tensor) -> (!tvm_ffi_abi.tensor, i1)
      %last = scf.if %second_status -> (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>) {
        %pure = tvm_ffi_monad.pure %second : (!tvm_ffi_abi.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
        scf.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
      } else {
        %failed = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
        scf.yield %failed : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
      }
      tvm_ffi_monad.yield %last : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
    }
    tvm_ffi_monad.yield %continued : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  }
  %return_ok_1, %return_value_1 = tvm_ffi_monad.unpack %output : (!tvm_ffi_monad.result<!tvm_ffi_abi.tensor>) -> (i1, !tvm_ffi_api.any)
  %return_flag_1 = tvm_ffi_abi.to %return_ok_1 : i1 -> !tvm_ffi_abi.bool
  tvm_ffi_api.return %return_flag_1, %return_value_1 : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// A None payload preserves the status dependency for independent side effects.
tvm_ffi_api.func @none_continuation(%boxed_ready: !tvm_ffi_abi.bool, %source: !tvm_ffi_abi.tensor, %destination: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.none {
  %none = tvm_ffi_abi.constant.none
  %ready = tvm_ffi_abi.get %boxed_ready : !tvm_ffi_abi.bool -> i1
  %input = scf.if %ready -> (!tvm_ffi_monad.result<!tvm_ffi_abi.none>) {
    %pure = tvm_ffi_monad.pure %none : (!tvm_ffi_abi.none) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    scf.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  } else {
    %fail = tvm_ffi_monad.fail : !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    scf.yield %fail : !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  }
  %output = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none> {
  ^bb0(%payload: !tvm_ffi_abi.none):
    tvm_ffi_api.tensor.copy_ %destination, %source : !tvm_ffi_abi.tensor
    %pure = tvm_ffi_monad.pure %payload : (!tvm_ffi_abi.none) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    tvm_ffi_monad.yield %pure : !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  }
  %return_ok_2, %return_value_2 = tvm_ffi_monad.unpack %output : (!tvm_ffi_monad.result<!tvm_ffi_abi.none>) -> (i1, !tvm_ffi_api.any)
  %return_flag_2 = tvm_ffi_abi.to %return_ok_2 : i1 -> !tvm_ffi_abi.bool
  tvm_ffi_api.return %return_flag_2, %return_value_2 : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

func.func private @lookup() -> (!tvm_ffi_abi.function, i1)
func.func private @checked_call(!tvm_ffi_abi.function, !tvm_ffi_abi.tensor) -> (!tvm_ffi_abi.tensor, i1)
