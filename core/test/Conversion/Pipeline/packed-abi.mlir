// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// Scalar results are boxed by the graph and forwarded into the packed result.
// CHECK-LABEL: llvm.func @make_int(
// CHECK-SAME: %[[SCALAR_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(42 : i64)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[PREFIX:[a-zA-Z0-9_]+]][2]
// CHECK: llvm.store %[[RESULT]], %[[SCALAR_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[RESULT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @make_int(%[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
// CHECK: llvm.store %[[BOXED]], %[[RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_int() -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 42

  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// Multi-results cross the ABI through an Array, rather than a native tuple.
// CHECK-LABEL: llvm.func @multi_return_three_tensors(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Y:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Z:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[X]], %[[SLOT_X:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[Y]], %[[SLOT_Y:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[Z]], %[[SLOT_Z:[a-zA-Z0-9_]+]]
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_three_tensors(
// CHECK-SAME: %[[MULTI_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[MULTI_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[MULTI_NARGS:[a-zA-Z0-9_]+]]: i32, %[[MULTI_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[ARRAY_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_ffi.Array
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32)
// CHECK: %[[INPUT_X:[a-zA-Z0-9_]+]] = llvm.load %[[MULTI_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INPUT_Y_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[MULTI_ARGS]][1]
// CHECK: %[[INPUT_Y:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Y_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INPUT_Z_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[MULTI_ARGS]][2]
// CHECK: %[[INPUT_Z:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Z_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @multi_return_three_tensors(%[[INPUT_X]], %[[INPUT_Y]], %[[INPUT_Z]], %[[OUT_X:[a-zA-Z0-9_]+]], %[[OUT_Y:[a-zA-Z0-9_]+]], %[[OUT_Z:[a-zA-Z0-9_]+]])
// CHECK: %[[CALL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[OUTPUT_X:[a-zA-Z0-9_]+]] = llvm.load %[[OUT_X]]
// CHECK: llvm.store %[[OUTPUT_X]], %[[CALL_ARGS]]
// CHECK: %[[ELEMENT_Y_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CALL_ARGS]][1]
// CHECK: %[[OUTPUT_Y:[a-zA-Z0-9_]+]] = llvm.load %[[OUT_Y]]
// CHECK: llvm.store %[[OUTPUT_Y]], %[[ELEMENT_Y_SLOT]]
// CHECK: %[[ELEMENT_Z_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CALL_ARGS]][2]
// CHECK: %[[OUTPUT_Z:[a-zA-Z0-9_]+]] = llvm.load %[[OUT_Z]]
// CHECK: llvm.store %[[OUTPUT_Z]], %[[ELEMENT_Z_SLOT]]
// CHECK: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE]], %[[CALL_ARGS]], %[[COUNT]], %[[ARRAY_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARRAY]], %[[MULTI_RESULT]]
tvm_ffi.func @multi_return_three_tensors(%x: !tvm_ffi.tensor,
    %y: !tvm_ffi.tensor, %z: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
    attributes {emit_tvm_ffi_abi} {

  %returned:3 = tvm_ffi.monad.pure %x, %y, %z : (!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
  tvm_ffi.return %returned#0, %returned#1, %returned#2 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
