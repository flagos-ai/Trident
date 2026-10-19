// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.'

// A failed Result skips a continuation with an independent side effect.
// CHECK-LABEL: llvm.func @short_circuit(
// CHECK-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1, %[[COUNTER:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.cond_br %[[READY]], ^[[YES:bb[0-9]+]], ^[[NO:bb[0-9]+]]
// CHECK: ^[[YES]]:
// CHECK: llvm.br ^[[MERGE:bb[0-9]+]](%[[SUCCESS:[a-zA-Z0-9_]+]] : !llvm.struct<(struct<(i32, i32, i64)>, i1)>)
// CHECK: ^[[NO]]:
// CHECK-NOT: llvm.store
// CHECK: llvm.br ^[[MERGE]](%[[FAILED:[a-zA-Z0-9_]+]] : !llvm.struct<(struct<(i32, i32, i64)>, i1)>)
// CHECK: ^[[MERGE]](%[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>):
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[INPUT]][1]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[INPUT]][0]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.select %[[OK]], %[[VALUE]], %[[ZERO]]
// CHECK: llvm.cond_br %[[OK]], ^[[VALIDATE:bb[0-9]+]], ^[[FAIL:bb[0-9]+]]
// CHECK: ^[[VALIDATE]]:
// CHECK: %[[TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[PAYLOAD]][0]
// CHECK: llvm.cond_br %[[TAG_OK:[a-zA-Z0-9_]+]], ^[[CONTINUE:bb[0-9]+]], ^[[ABORT:bb[0-9]+]]
// CHECK: ^[[ABORT]]:
// CHECK: llvm.call @abort()
// CHECK: llvm.unreachable
// CHECK: ^[[CONTINUE]]:
// CHECK: %[[OLD:[a-zA-Z0-9_]+]] = llvm.load %[[COUNTER]]
// CHECK: %[[NEW:[a-zA-Z0-9_]+]] = llvm.add %[[OLD]], %[[ONE:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[NEW]], %[[COUNTER]]
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[EMPTY:[a-zA-Z0-9_]+]][0]
// CHECK: %[[RETURNED:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[TRUE:[a-zA-Z0-9_]+]], %[[PURE]][1]
// CHECK: llvm.br ^[[RETURN:bb[0-9]+]](%[[RETURNED]] : !llvm.struct<(struct<(i32, i32, i64)>, i1)>)
// CHECK: ^[[FAIL]]:
// CHECK-NOT: llvm.store
// CHECK: llvm.br ^[[RETURN]](%[[FAILURE:[a-zA-Z0-9_]+]] : !llvm.struct<(struct<(i32, i32, i64)>, i1)>)
// CHECK: ^[[RETURN]](%[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>):
// CHECK: %[[FINAL_OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[FINAL_VALUE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: %[[FINAL_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.select %[[FINAL_OK]], %[[FINAL_VALUE]], %[[ZERO]]
// CHECK: llvm.cond_br %[[FINAL_OK]], ^[[CHECK_RETURN:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[CHECK_RETURN]]:
// CHECK: %[[FINAL_TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[FINAL_PAYLOAD]][0]
// CHECK: llvm.cond_br %[[RETURN_TAG_OK:[a-zA-Z0-9_]+]], ^[[STORE:bb[0-9]+]], ^[[ABORT]]
// CHECK: ^[[STORE]]:
// CHECK: llvm.store %[[FINAL_PAYLOAD]], %[[SLOT]]
// CHECK: llvm.br ^[[EXIT]]
// CHECK: ^[[EXIT]]:
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @short_circuit(%ready: i1, %counter: !llvm.ptr) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %input = scf.if %ready -> !tvm_ffi.monad.result<!tvm_ffi.int> {
    %seven = arith.constant 7 : i64
    %seven_boxed = tvm_ffi.to %seven : i64 -> !tvm_ffi.int
    %pure = tvm_ffi.monad.pure %seven_boxed {tvm_ffi.cleanup} : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield {tvm_ffi.retain} %pure : !tvm_ffi.monad.result<!tvm_ffi.int>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.int>
    scf.yield {tvm_ffi.retain} %failed : !tvm_ffi.monad.result<!tvm_ffi.int>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  ^bb0(%value_boxed: !tvm_ffi.int):
    %old = llvm.load %counter : !llvm.ptr -> i64
    %one = arith.constant 1 : i64
    %new = arith.addi %old, %one : i64
    llvm.store %new, %counter : i64, !llvm.ptr
    %result = tvm_ffi.monad.pure %value_boxed {tvm_ffi.cleanup} : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    tvm_ffi.monad.yield {tvm_ffi.retain} %result : !tvm_ffi.monad.result<!tvm_ffi.int>
  } {tvm_ffi.cleanup}
  tvm_ffi.return {tvm_ffi.retain} %output : !tvm_ffi.monad.result<!tvm_ffi.int>
}
