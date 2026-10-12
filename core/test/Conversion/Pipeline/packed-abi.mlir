// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// Scalar results are boxed by the graph and forwarded into the packed result.
// CHECK-LABEL: llvm.func @make_int()
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(42 : i64)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[PREFIX:[a-zA-Z0-9_]+]][2]
// CHECK: llvm.return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[RESULT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = llvm.call @make_int()
// CHECK: llvm.store %[[BOXED]], %[[RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_int() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 42
  tvm_ffi.return %value : !tvm_ffi.int
}

// Multi-results cross the ABI through an Array, rather than a native tuple.
// CHECK-LABEL: llvm.func @multi_return_three_tensors(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Y:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Z:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[X]], %[[UNDEF:[a-zA-Z0-9_]+]][0]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[Y]], %[[FIRST]][1]
// CHECK: %[[THIRD:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[Z]], %[[SECOND]][2]
// CHECK: llvm.return %[[THIRD]]
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_three_tensors(
// CHECK-SAME: %[[MULTI_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[MULTI_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[MULTI_NARGS:[a-zA-Z0-9_]+]]: i32, %[[MULTI_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[ARRAY_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_ffi.Array
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32)
// CHECK: %[[INPUT_X:[a-zA-Z0-9_]+]] = llvm.load %[[MULTI_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INPUT_Y_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[MULTI_ARGS]][1]
// CHECK: %[[INPUT_Y:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Y_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INPUT_Z_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[MULTI_ARGS]][2]
// CHECK: %[[INPUT_Z:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Z_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[RETURNED:[a-zA-Z0-9_]+]] = llvm.call @multi_return_three_tensors(%[[INPUT_X]], %[[INPUT_Y]], %[[INPUT_Z]])
// CHECK: %[[OUTPUT_X:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RETURNED]][0]
// CHECK: %[[OUTPUT_Y:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RETURNED]][1]
// CHECK: %[[OUTPUT_Z:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[RETURNED]][2]
// CHECK: %[[ELEMENTS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[OUTPUT_X]], %[[ELEMENTS]]
// CHECK: %[[ELEMENT_Y_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ELEMENTS]][1]
// CHECK: llvm.store %[[OUTPUT_Y]], %[[ELEMENT_Y_SLOT]]
// CHECK: %[[ELEMENT_Z_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ELEMENTS]][2]
// CHECK: llvm.store %[[OUTPUT_Z]], %[[ELEMENT_Z_SLOT]]
// CHECK: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE]], %[[CALL_ARGS:[a-zA-Z0-9_]+]], %[[COUNT]], %[[ARRAY_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARRAY]], %[[MULTI_RESULT]]
tvm_ffi.func @multi_return_three_tensors(%x: !tvm_ffi.tensor,
    %y: !tvm_ffi.tensor, %z: !tvm_ffi.tensor)
    -> (!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor)
    attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %x, %y, %z : !tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor
}
