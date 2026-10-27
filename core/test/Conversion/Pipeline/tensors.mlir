// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.' --implicit-check-not='dlpack.' --implicit-check-not='unrealized_conversion_cast'

// The local LowerTensorMetadata and DLPackToLLVM fixtures own the metadata field
// matrix. This integration follows one dynamic index from a semantic tensor
// through object extraction, the DLTensor layout and the native return value.
// CHECK-LABEL: llvm.func @tensor_size(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OBJECT]][24]
// CHECK: %[[DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[DLTENSOR_PTR]]
// CHECK: %[[SIZES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DLTENSOR]][4]
// CHECK: %[[SIZE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZES]][%[[INDEX]]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK: llvm.return %[[SIZE]] : i64
func.func @tensor_size(%tensor: !tvm_ffi.tensor, %index: i64) -> i64 {
  %size = tvm_ffi.tensor.size %tensor[%index] : !tvm_ffi.tensor
  return %size : i64
}

// -----

// Tensor literals use one stable CPU-staging path for both splat and
// non-splat attributes before copying the tensor to the current CUDA device.

// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_literal_nonsplat(
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob(%[[NONSPLAT_DATA:[a-zA-Z0-9_]+]], %[[NONSPLAT_RANK:[a-zA-Z0-9_]+]], %[[NONSPLAT_SHAPE:[a-zA-Z0-9_]+]], %[[NONSPLAT_STRIDES:[a-zA-Z0-9_]+]], %[[NONSPLAT_OFFSET:[a-zA-Z0-9_]+]], %[[NONSPLAT_DTYPE:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_DEVICE:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_INDEX:[a-zA-Z0-9_]+]], %[[NONSPLAT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[NONSPLAT_CPU_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[NONSPLAT_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @aoti_torch_copy_(%[[NONSPLAT_CUDA_TENSOR:[a-zA-Z0-9_]+]], %[[NONSPLAT_CPU_TENSOR]], %[[NONSPLAT_NON_BLOCKING:[a-zA-Z0-9_]+]])
// CHECK: llvm.call @aoti_torch_delete_tensor_object(%[[NONSPLAT_CPU_TENSOR]])

// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_literal_splat(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUTPUT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr)
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
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.br ^[[CHECKED:bb[0-9]+]](%[[RESULT]] : !llvm.struct<(i32, i32, i64)>)
// CHECK: ^[[CHECKED]](%[[PAYLOAD:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>):
// CHECK: llvm.cond_br %[[SUCCESS:[a-zA-Z0-9_]+]], ^[[STORE:bb[0-9]+]](%[[PAYLOAD]], %[[CODE:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, i32)
// CHECK: ^[[STORE]](%[[OUTPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[STORE_CODE:[a-zA-Z0-9_]+]]: i32):
// CHECK-NEXT: llvm.store %[[OUTPUT]], %[[OUTPUT_SLOT]]
// The packed entry stores the checked payload directly into its output slot.
tvm_ffi.func @tensor_literal_nonsplat() -> !tvm_ffi.c.tensor {
  %0 = tvm_ffi.tensor.literal dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> : !tvm_ffi.c.tensor

  // The marked return retains its payload before local cleanup.
  %return_value_1 = tvm_ffi.upcast %0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %return_flag_1 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}

// The packed entry stores the checked payload directly into its output slot.
tvm_ffi.func @tensor_literal_splat() -> !tvm_ffi.c.tensor {
  %0 = tvm_ffi.tensor.literal dense<1.250000e+00> : tensor<2x3xf32> : !tvm_ffi.c.tensor

  // The marked return retains its payload before local cleanup.
  %return_value_2 = tvm_ffi.upcast %0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %return_flag_2 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.any
}
