// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.'

// A failed native Result skips a continuation with an independent side effect.
// CHECK-LABEL: llvm.func @short_circuit(
// CHECK-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1, %[[COUNTER:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: llvm.cond_br %[[READY]], ^[[MERGE:bb[0-9]+]](%[[SEVEN:[a-zA-Z0-9_]+]], %[[YES:[a-zA-Z0-9_]+]] : i64, i1), ^[[MERGE]](%[[POISON:[a-zA-Z0-9_]+]], %[[NO:[a-zA-Z0-9_]+]] : i64, i1)
// CHECK: ^[[MERGE]](%[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[OK:[a-zA-Z0-9_]+]]: i1):
// CHECK: llvm.cond_br %[[OK]], ^[[CONTINUE:bb[0-9]+]], ^[[FAIL:bb[0-9]+]]
// CHECK: ^[[CONTINUE]]:
// CHECK: %[[OLD:[a-zA-Z0-9_]+]] = llvm.load %[[COUNTER]]
// CHECK: %[[NEW:[a-zA-Z0-9_]+]] = llvm.add %[[OLD]], %[[ONE:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[NEW]], %[[COUNTER]]
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[EMPTY:[a-zA-Z0-9_]+]][0]
// CHECK: %[[RETURNED:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[TRUE:[a-zA-Z0-9_]+]], %[[PURE]][1]
// CHECK: llvm.br ^[[RETURN:bb[0-9]+]](%[[RETURNED]] : !llvm.struct<(i64, i1)>)
// CHECK: ^[[FAIL]]:
// CHECK-NOT: llvm.store
// CHECK: llvm.br ^[[RETURN]](%[[FAILURE:[a-zA-Z0-9_]+]] : !llvm.struct<(i64, i1)>)
// CHECK: ^[[RETURN]](%[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(i64, i1)>):
// CHECK: %[[FINAL_OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[FINAL_VALUE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: llvm.cond_br %[[FINAL_OK]], ^[[STORE:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[STORE]]:
// CHECK: llvm.store %[[FINAL_VALUE]], %[[SLOT]]
// CHECK: llvm.br ^[[EXIT]]
// CHECK: ^[[EXIT]]:
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @short_circuit(%ready: i1, %counter: !llvm.ptr) -> !tvm_ffi.monad.result<i64> {
  %input = scf.if %ready -> !tvm_ffi.monad.result<i64> {
    %seven = arith.constant 7 : i64
    %pure = tvm_ffi.monad.pure %seven : (i64) -> !tvm_ffi.monad.result<i64>
    scf.yield %pure : !tvm_ffi.monad.result<i64>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
    scf.yield %failed : !tvm_ffi.monad.result<i64>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  ^bb0(%value: i64):
    %old = llvm.load %counter : !llvm.ptr -> i64
    %one = arith.constant 1 : i64
    %new = arith.addi %old, %one : i64
    llvm.store %new, %counter : i64, !llvm.ptr
    %result = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    tvm_ffi.monad.yield %result : !tvm_ffi.monad.result<i64>
  }
  tvm_ffi.return %output : !tvm_ffi.monad.result<i64>
}
