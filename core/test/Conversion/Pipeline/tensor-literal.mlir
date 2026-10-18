//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Keep input/check order: ABI lowering inserts helper declarations and wrappers
// whose output order is checked together with the original functions.
// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// Tensor literals use one stable CPU-staging path for both splat and
// non-splat attributes before copying the tensor to the current CUDA device.

// CHECK-LABEL: llvm.func @tensor_literal_nonsplat(
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob(%[[NONSPLAT_DATA:[a-zA-Z0-9_]+]], %[[NONSPLAT_RANK:[a-zA-Z0-9_]+]], %[[NONSPLAT_SHAPE:[a-zA-Z0-9_]+]], %[[NONSPLAT_STRIDES:[a-zA-Z0-9_]+]], %[[NONSPLAT_OFFSET:[a-zA-Z0-9_]+]], %[[NONSPLAT_DTYPE:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_DEVICE:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_INDEX:[a-zA-Z0-9_]+]], %[[NONSPLAT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[NONSPLAT_CPU_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[NONSPLAT_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @aoti_torch_copy_(%[[NONSPLAT_CUDA_TENSOR:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_TENSOR]], %[[NONSPLAT_NON_BLOCKING:[a-zA-Z0-9_]+]])
// CHECK: llvm.call @aoti_torch_delete_tensor_object(%[[NONSPLAT_CPU_TENSOR]])
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_literal_nonsplat(
// CHECK: llvm.call @tensor_literal_nonsplat(%[[SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @tensor_literal_splat(
// CHECK-SAME: %[[OUTPUT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.runtime.tensor_to_tvm_ffi_object : !llvm.ptr
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1.250000e+00 : f32)
// CHECK: %[[ELEMENTS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(6 : i64)
// CHECK: %[[RANK:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64)
// CHECK: %[[SHAPE_DATA:[a-zA-Z0-9_]+]] = llvm.alloca %[[RANK]] x i64
// CHECK: %[[STRIDES:[a-zA-Z0-9_]+]] = llvm.alloca %[[RANK]] x i64
// CHECK: %[[LITERAL_DATA:[a-zA-Z0-9_]+]] = llvm.alloca %[[ELEMENTS]] x f32
// CHECK: llvm.store %[[VALUE]], %[[LITERAL_DATA]] : f32, !llvm.ptr
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob(%[[LITERAL_DATA]], %[[RANK]], %[[SHAPE_DATA]], %[[STRIDES]], %[[OFFSET:[a-zA-Z0-9_]+]], %[[DTYPE:[a-zA-Z0-9_]+]], %[[CPU_DEVICE:[a-zA-Z0-9_]+]], %[[CPU_INDEX:[a-zA-Z0-9_]+]], %[[CPU_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[CPU_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[CPU_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @aoti_torch_empty_strided(%[[RANK]], %[[SHAPE_DATA]], %[[STRIDES]], %[[DTYPE]], %[[CUDA_DEVICE:[a-zA-Z0-9_]+]], %[[CUDA_INDEX:[a-zA-Z0-9_]+]], %[[CUDA_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[CUDA_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[CUDA_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @aoti_torch_copy_(%[[CUDA_TENSOR]], %[[CPU_TENSOR]], %[[NON_BLOCKING:[a-zA-Z0-9_]+]])
// CHECK: llvm.call @aoti_torch_delete_tensor_object(%[[CPU_TENSOR]])
// CHECK: %[[TENSOR_BITS:[a-zA-Z0-9_]+]] = llvm.ptrtoint %[[CUDA_TENSOR]] : !llvm.ptr to i64
// CHECK: llvm.store %[[TENSOR_BITS]], %[[TENSOR_PAYLOAD:[a-zA-Z0-9_]+]] : i64, !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[SELECTED:[a-zA-Z0-9_]+]] = llvm.select %[[OK:[a-zA-Z0-9_]+]], %[[RESULT]], %[[NONE:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[SELECTED]], %[[OUTPUT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_literal_splat(
// CHECK: llvm.call @tensor_literal_splat(%[[SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr) -> i32
// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
tvm_ffi.func @tensor_literal_nonsplat() -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {emit_tvm_ffi_abi} {
  %0 = tvm_ffi.tensor.literal dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> : !tvm_ffi.tensor

  %returned = tvm_ffi.monad.pure %0 : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
tvm_ffi.func @tensor_literal_splat() -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {emit_tvm_ffi_abi} {
  %0 = tvm_ffi.tensor.literal dense<1.250000e+00> : tensor<2x3xf32> : !tvm_ffi.tensor

  %returned = tvm_ffi.monad.pure %0 : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
