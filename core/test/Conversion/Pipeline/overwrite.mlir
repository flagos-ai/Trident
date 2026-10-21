//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Keep input/check order: ABI lowering inserts helper declarations and wrappers
// whose output order is checked together with the original functions.
// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// The in-place copy keeps the destination as the first tensor argument to the
// runtime copy function and passes the source as the second argument.

// CHECK-LABEL: llvm.func @overwrite(
// CHECK-SAME: %[[SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi} {
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGUMENTS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @__tvm_ffi_overwrite(
// CHECK-SAME: %[[WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAPPER_SOURCE:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAPPER_DESTINATION:[a-zA-Z0-9_]+]] = llvm.load %[[DEST_PTR:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.call @overwrite(%[[WRAPPER_SOURCE]], %[[WRAPPER_DESTINATION]], %[[RESULT_SLOT]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.ptr) -> i32

tvm_ffi.func @overwrite(%source: !tvm_ffi.c.tensor,
                         %destination: !tvm_ffi.c.tensor) -> !tvm_ffi.none
    attributes {emit_tvm_ffi_abi} {
  tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.c.tensor

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  %return_ok_1, %return_value_1 = tvm_ffi.monad.unpack %returned : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  %return_flag_1 = tvm_ffi.to %return_ok_1 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}
