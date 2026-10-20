//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Keep input/check order: ABI lowering inserts helper declarations and wrappers
// whose output order is checked together with the original functions.
// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// The list-valued dimensions operand must be constructed as an FFI Array
// and forwarded to the reduction call. Its public wrapper unpacks the input.

// CHECK-LABEL: llvm.func @all_dims(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[OUTPUT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi} {
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.all.dims : !llvm.ptr
// The first dispatch constructs the FFI Array for the list of dimensions.
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE:[a-zA-Z0-9_]+]], %[[ARRAY_ARGS:[a-zA-Z0-9_]+]], %[[ARRAY_ARG_COUNT:[a-zA-Z0-9_]+]], %[[ARRAY_RETURN_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[DIMS:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_RETURN_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// The callee is resolved once at load time, so the dispatch reads the cached
// handle rather than calling TVMFFIFunctionGetGlobal per invocation.
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[SOURCE_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[ARG0]], %[[SOURCE_ARGS]]
// CHECK: %[[DIMS_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SOURCE_ARGS]][1]
// CHECK: llvm.store %[[DIMS]], %[[DIMS_SLOT]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[RET]], %[[OUTPUT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_all_dims(
// CHECK-SAME: %[[WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAPPER_INPUT:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAPPER_STATUS:[a-zA-Z0-9_]+]] = llvm.call @all_dims(%[[WRAPPER_INPUT]], %[[WRAPPER_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.ptr) -> i32
// CHECK: %[[WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAPPER_RET]], %[[WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr

tvm_ffi.func @all_dims(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %int1 = tvm_ffi.constant.int 1
  %int0 = tvm_ffi.constant.int 0
  %call_dims_handle = tvm_ffi.FunctionGetGlobal "ffi.Array" {tvm_ffi.cleanup} : !tvm_ffi.object
  %dims = tvm_ffi.FunctionCall %call_dims_handle(%int1, %int0) {tvm_ffi.cleanup} : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.c.array
  %false = tvm_ffi.constant.bool false
  %call_result_handle = tvm_ffi.FunctionGetGlobal "trident.aten.all.dims" {tvm_ffi.cleanup} : !tvm_ffi.object
  %result = tvm_ffi.FunctionCall %call_result_handle(%arg0, %dims, %false) {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.c.array, !tvm_ffi.bool) -> !tvm_ffi.tensor

  // The marked return retains its payload before local cleanup.
  %return_value_1 = tvm_ffi.upcast %result : !tvm_ffi.tensor -> !tvm_ffi.any
  %return_flag_1 = tvm_ffi.constant.bool true
  tvm_ffi.return {tvm_ffi.retain} %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}
