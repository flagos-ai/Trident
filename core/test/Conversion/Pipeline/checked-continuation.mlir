// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.monad.'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.'

// Result construction is covered by lower-monad.mlir. Here each continuation
// must gate the next call, even when it does not consume the preceding payload.
// CHECK-LABEL: tvm_ffi.func @checked_continuation(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[LOOKUP:[a-zA-Z0-9_]+]]:2 = func.call @lookup()
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = scf.if %[[LOOKUP]]#1
// CHECK: %[[READY_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[READY]]
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[READY_PARTS]]#0
// CHECK-NEXT: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if %[[LOOKUP_OK]]
// CHECK-NEXT: %[[CALLEE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[READY_PARTS]]#1
// CHECK-NEXT: %[[FIRST:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// CHECK-NEXT: %[[FIRST_RESULT:[a-zA-Z0-9_]+]] = scf.if %[[FIRST]]#1
// CHECK: %[[FIRST_PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[FIRST_RESULT]]
// CHECK: %[[FIRST_OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[FIRST_PARTS]]#0
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = scf.if %[[FIRST_OK]]
// CHECK-NEXT: %[[UNUSED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[FIRST_PARTS]]#1
// CHECK-NEXT: %[[SECOND:[a-zA-Z0-9_]+]]:2 = func.call @checked_call(%[[CALLEE]], %[[TENSOR]])
// CHECK-NEXT: %[[SECOND_RESULT:[a-zA-Z0-9_]+]] = scf.if %[[SECOND]]#1
// CHECK: scf.yield %[[SECOND_RESULT]]
// CHECK: scf.yield %[[INNER]]
// CHECK-NOT: func.call
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[OUTER]]
// CHECK: tvm_ffi.return %[[OUTPUT]]#0, %[[OUTPUT]]#1

// A None payload must still gate a payload-independent mutation.
// CHECK-LABEL: tvm_ffi.func @none_continuation(
// CHECK-SAME: %[[BOXED:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[SOURCE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[DEST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[BOXED]]
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: %[[PARTS:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[INPUT]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[PARTS]]#0
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK-NEXT: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PARTS]]#1
// CHECK-NEXT: tvm_ffi.tensor.copy_ %[[DEST]], %[[SOURCE]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.tensor.copy_
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]]:2 = tvm_ffi.tuple.unconstruct %[[RESULT]]
// CHECK: tvm_ffi.return %[[OUTPUT]]#0, %[[OUTPUT]]#1

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

tvm_ffi.func @checked_continuation(%tensor: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor {
  %handle, %lookup = func.call @lookup() : () -> (!tvm_ffi.c.function, i1)
  %ready = scf.if %lookup -> (!tvm_ffi.monad.result<!tvm_ffi.c.function>) {
    %pure = tvm_ffi.monad.pure %handle : (!tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.function>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.function>
    scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.function>
  }
  %output = tvm_ffi.monad.bind %ready : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  ^bb0(%callee: !tvm_ffi.c.function):
    %value, %status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
    %first = scf.if %status -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
      %pure = tvm_ffi.monad.pure %value : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    } else {
      %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    }
    %continued = tvm_ffi.monad.bind %first : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
    ^bb0(%unused: !tvm_ffi.c.tensor):
      // Deliberately independent of the first payload: effects still short-circuit.
      %second, %second_status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
      %last = scf.if %second_status -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
        %pure = tvm_ffi.monad.pure %second : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
        scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      } else {
        %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
        scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      }
      tvm_ffi.monad.yield %last : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    }
    tvm_ffi.monad.yield %continued : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  }
  %return_ok_1, %return_value_1 = tvm_ffi.monad.unpack %output : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> (i1, !tvm_ffi.any)
  %return_flag_1 = tvm_ffi.to %return_ok_1 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}

// A None payload preserves the status dependency for independent side effects.
tvm_ffi.func @none_continuation(%boxed_ready: !tvm_ffi.bool, %source: !tvm_ffi.c.tensor, %destination: !tvm_ffi.c.tensor) -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %ready = tvm_ffi.get %boxed_ready : !tvm_ffi.bool -> i1
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
    %pure = tvm_ffi.monad.pure %none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %fail : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%payload: !tvm_ffi.none):
    tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.c.tensor
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %return_ok_2, %return_value_2 = tvm_ffi.monad.unpack %output : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  %return_flag_2 = tvm_ffi.to %return_ok_2 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.any
}

func.func private @lookup() -> (!tvm_ffi.c.function, i1)
func.func private @checked_call(!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
