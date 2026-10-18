// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @attributes(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: i64 {test.arg}, %[[STATUS:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64> {test.result}, !tvm_ffi.monad.result<!tvm_ffi.none>)
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ARG]], %[[STATUS]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[UNIT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]], %[[TRUE]] : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: tvm_ffi.return %[[FIRST]], %[[UNIT]]
tvm_ffi.func @attributes(%input: !tvm_ffi.monad.result<i64> {test.arg}) -> (!tvm_ffi.monad.result<i64> {test.result}, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %unit_none = tvm_ffi.constant.none
  %unit = tvm_ffi.monad.pure %unit_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>

  tvm_ffi.return %input, %unit : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
}

// Native handles require null pointer placeholders, not boxed Any storage.
// CHECK-LABEL: tvm_ffi.func @failed_handle()
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK: %[[NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NULL]] : !llvm.ptr to !tvm_ffi.function
// CHECK: %[[FAILURE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[HANDLE]], %[[FALSE]]
// CHECK: tvm_ffi.return %[[FAILURE]]
tvm_ffi.func @failed_handle() -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>

  tvm_ffi.return %failed : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// CHECK-LABEL: tvm_ffi.func @failure() -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK: %[[POISON:[a-zA-Z0-9_]+]] = ub.poison : i64
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ZERO]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.tensor
// CHECK: %[[NONE_ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE_ZERO]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.none
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[POISON]], %[[FALSE]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ABI]], %[[FALSE]] : !tvm_ffi.tensor, i1 to !tvm_ffi.monad.result<!tvm_ffi.tensor>
// CHECK: %[[UNIT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]], %[[FALSE]] : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: tvm_ffi.return %[[FIRST]], %[[SECOND]], %[[UNIT]]
tvm_ffi.func @failure() -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %failed:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>

  tvm_ffi.return %failed#0, %failed#1, %failed#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @mixed_return(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[FIRST:[a-zA-Z0-9_]+]]: i1, %[[SECOND:[a-zA-Z0-9_]+]]: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]]:2 = scf.if %[[FIRST]] -> (i64, i1)
// CHECK: %[[UNIT:[a-zA-Z0-9_]+]]:2 = scf.if %[[SECOND]] -> (!tvm_ffi.none, i1)
// CHECK: }
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[PLAIN:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VALUE]], %[[TRUE]] : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[WRAPPED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RESULT]]#0, %[[RESULT]]#1 : i64, i1 to !tvm_ffi.monad.result<i64>
// CHECK: %[[UNIT_RESULT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[UNIT]]#0, %[[UNIT]]#1 : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: tvm_ffi.return %[[PLAIN]], %[[WRAPPED]], %[[UNIT_RESULT]]
tvm_ffi.func @mixed_return(%value: i64, %first: i1, %second: i1) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %result = scf.if %first -> (!tvm_ffi.monad.result<i64>) {
    %result_pure = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
    scf.yield %result_pure : !tvm_ffi.monad.result<i64>
  } else {
    %result_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>
    scf.yield %result_fail : !tvm_ffi.monad.result<i64>
  }
  %unit = scf.if %second -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
    %unit_pure_none = tvm_ffi.constant.none
    %unit_pure = tvm_ffi.monad.pure %unit_pure_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %unit_pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %unit_fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %unit_fail : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %returned = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>

  tvm_ffi.return %returned, %result, %unit : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
}
