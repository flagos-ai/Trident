//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// An explicit clone call is preserved through canonicalization and
// reaches the name-based TVM FFI dispatch path with its memory format operand.
// In particular, the contiguous-memory-format clone must not fold to its input:
// test/e2e/test_storage.py checks distinct allocations for contiguous
// and preserve-format clones through the public frontend.

// CHECK-LABEL: llvm.func @clone(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[OUTPUT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi} {
// The callee is resolved once at load time, so the dispatch reads the cached
// handle rather than calling TVMFFIFunctionGetGlobal per invocation. The
// module owns the cached reference.
// CHECK-NOT: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.clone : !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARG0]], %[[ARGS]]
// CHECK: %[[FORMAT_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[FORMAT:[a-zA-Z0-9_]+]], %[[FORMAT_SLOT]]
// CHECK: %[[RET_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// Automatic ownership releases the local handle before exporting the payload.
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: llvm.store %[[RET]], %[[OUTPUT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_clone(
// CHECK-SAME: %[[WRAP_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAP_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[WRAP_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_STATUS:[a-zA-Z0-9_]+]] = llvm.call @clone(%[[WRAP_ARG]], %[[WRAP_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.ptr) -> i32
// CHECK: %[[WRAP_RET:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAP_RET]], %[[WRAP_RESULT]]

tvm_ffi.func @clone(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %memory_format = tvm_ffi.constant.int 0
  %clone_handle = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function
  %clone = tvm_ffi.FunctionCall %clone_handle : !tvm_ffi.function(%arg0, %memory_format) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor

  // The return retains its payload before local cleanup.
  %return_value_1 = tvm_ffi.upcast %clone : !tvm_ffi.tensor -> !tvm_ffi.any
  %return_flag_1 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}
