// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s
// RUN: trident-core-opt %s --convert-to-llvm --verify-each | FileCheck %s --check-prefix=FULL --implicit-check-not='!tvm_ffi.'

// FULL-LABEL: llvm.func @construct(
// CHECK-LABEL: func.func @construct(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64) -> !llvm.struct<(i64, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = llvm.mlir.poison : !llvm.struct<(i64, i1)>
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[EMPTY]][0]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[OK]], %[[PAYLOAD]][1]
// CHECK: return %[[RESULT]]
func.func @construct(%value: i64) -> !tvm_ffi.monad.result<i64> {
  %result = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
  return %result : !tvm_ffi.monad.result<i64>
}

// FULL-LABEL: llvm.func @construct_multiple(
// CHECK-LABEL: func.func @construct_multiple(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[NONE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[HANDLE:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// CHECK: %[[VALUE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[VALUE_EMPTY:[a-zA-Z0-9_]+]][0]
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[OK]], %[[VALUE_PAYLOAD]][1]
// CHECK: %[[NONE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NONE]], %[[NONE_EMPTY:[a-zA-Z0-9_]+]][0]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[OK]], %[[NONE_PAYLOAD]][1]
// CHECK: %[[HANDLE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[HANDLE]], %[[HANDLE_EMPTY:[a-zA-Z0-9_]+]][0]
// CHECK: %[[THIRD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[OK]], %[[HANDLE_PAYLOAD]][1]
// CHECK: return %[[FIRST]], %[[SECOND]], %[[THIRD]]
func.func @construct_multiple(%value: i64, %none: !tvm_ffi.none, %object: !tvm_ffi.object) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.object>) {
  %results:3 = tvm_ffi.monad.pure %value, %none, %object : (i64, !tvm_ffi.none, !tvm_ffi.object) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.object>)
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>, !tvm_ffi.monad.result<!tvm_ffi.object>
}

// FULL-LABEL: llvm.func @failures(
// CHECK-LABEL: func.func @failures(
// CHECK: %[[NO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(false) : i1
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = llvm.mlir.poison : i64
// CHECK: %[[NATIVE_EMPTY:[a-zA-Z0-9_]+]] = llvm.mlir.poison : !llvm.struct<(i64, i1)>
// CHECK: %[[NATIVE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NATIVE]], %[[NATIVE_EMPTY]][0]
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NO]], %[[NATIVE_PAYLOAD]][1]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[TENSOR_EMPTY:[a-zA-Z0-9_]+]] = llvm.mlir.poison : !llvm.struct<(struct<(i32, i32, i64)>, i1)>
// CHECK: %[[TENSOR_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[ZERO]], %[[TENSOR_EMPTY]][0]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NO]], %[[TENSOR_PAYLOAD]][1]
// CHECK: %[[NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[HANDLE_EMPTY:[a-zA-Z0-9_]+]] = llvm.mlir.poison : !llvm.struct<(ptr, i1)>
// CHECK: %[[HANDLE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NULL]], %[[HANDLE_EMPTY]][0]
// CHECK: %[[THIRD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[NO]], %[[HANDLE_PAYLOAD]][1]
// CHECK: return %[[FIRST]], %[[SECOND]], %[[THIRD]]
func.func @failures() -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.function>) {
  %results:3 = tvm_ffi.monad.fail : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.function>
  return %results#0, %results#1, %results#2 : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.function>
}

// FULL-LABEL: llvm.func @forward(
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i64, i1)>
// CHECK: cf.br ^[[BLOCK:bb[0-9]+]](%[[INPUT]] : !llvm.struct<(i64, i1)>)
// CHECK: ^[[BLOCK]](%[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i64, i1)>):
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @identity(%[[ARG]]) : (!llvm.struct<(i64, i1)>) -> !llvm.struct<(i64, i1)>
// CHECK: return %[[RESULT]]
func.func @forward(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  cf.br ^next(%input : !tvm_ffi.monad.result<i64>)
^next(%arg: !tvm_ffi.monad.result<i64>):
  %result = call @identity(%arg) : (!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64>
  return %result : !tvm_ffi.monad.result<i64>
}

func.func private @identity(!tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64>

// FULL-LABEL: llvm.func @unpack_any(
// CHECK-LABEL: func.func @unpack_any(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.select %[[OK]], %[[PAYLOAD]], %[[NONE]]
// CHECK: return %[[OK]], %[[VALUE]]
func.func @unpack_any(%result: !tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// FULL-LABEL: llvm.func @unpack_float(
// CHECK-LABEL: func.func @unpack_float(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(f64, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: return %[[OK]], %[[PAYLOAD]]
func.func @unpack_float(%result: !tvm_ffi.monad.result<f64>) -> (i1, f64) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<f64>) -> (i1, f64)
  return %ok, %value : i1, f64
}

// FULL-LABEL: llvm.func @unpack_function(
// CHECK-LABEL: func.func @unpack_function(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(ptr, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: return %[[OK]], %[[PAYLOAD]]
func.func @unpack_function(%result: !tvm_ffi.monad.result<!tvm_ffi.function>) -> (i1, !tvm_ffi.function) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.function>) -> (i1, !tvm_ffi.function)
  return %ok, %value : i1, !tvm_ffi.function
}

// FULL-LABEL: llvm.func @unpack_native(
// CHECK-LABEL: func.func @unpack_native(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(i64, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: return %[[OK]], %[[VALUE]]
func.func @unpack_native(%result: !tvm_ffi.monad.result<i64>) -> (i1, i64) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  return %ok, %value : i1, i64
}

// FULL-LABEL: llvm.func @unpack_none(
// CHECK-LABEL: func.func @unpack_none(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.select %[[OK]], %[[PAYLOAD]], %[[NONE]]
// CHECK: return %[[OK]], %[[VALUE]]
func.func @unpack_none(%result: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// FULL-LABEL: llvm.func @unpack_object(
// CHECK-LABEL: func.func @unpack_object(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(ptr, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: return %[[OK]], %[[PAYLOAD]]
func.func @unpack_object(%result: !tvm_ffi.monad.result<!tvm_ffi.object>) -> (i1, !tvm_ffi.object) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.object>) -> (i1, !tvm_ffi.object)
  return %ok, %value : i1, !tvm_ffi.object
}

// FULL-LABEL: llvm.func @unpack_tensor(
// CHECK-LABEL: func.func @unpack_tensor(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.select %[[OK]], %[[PAYLOAD]], %[[NONE]]
// CHECK: return %[[OK]], %[[VALUE]]
func.func @unpack_tensor(%result: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}

// FULL-LABEL: llvm.func @unpack_union(
// CHECK-LABEL: func.func @unpack_union(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !llvm.struct<(struct<(i32, i32, i64)>, i1)>
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RESULT]][0]
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.select %[[OK]], %[[PAYLOAD]], %[[NONE]]
// CHECK: return %[[OK]], %[[VALUE]]
func.func @unpack_union(%result: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.monad.unpack %result : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any)
  return %ok, %value : i1, !tvm_ffi.any
}
