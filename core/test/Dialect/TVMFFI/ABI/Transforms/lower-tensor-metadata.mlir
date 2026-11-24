//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --lower-tvm-ffi-abi-tensor-metadata --verify-each | FileCheck %s

// CHECK-LABEL: func.func @tensor_dtype_and_device(
// CHECK-SAME: %[[TENSOR1:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[OBJECT4:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR1]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR4:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT4]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[CODE:[a-zA-Z0-9_]+]], %[[BITS:[a-zA-Z0-9_]+]], %[[LANES:[a-zA-Z0-9_]+]] = dlpack.tensor.dtype %[[DLTENSOR4]] : !dlpack.dl_tensor
// CHECK: %[[OBJECT5:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR1]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR5:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT5]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]], %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = dlpack.tensor.device %[[DLTENSOR5]] : !dlpack.dl_tensor
// CHECK: return %[[CODE]], %[[BITS]], %[[LANES]], %[[DEVICE_TYPE]], %[[DEVICE_INDEX]]
func.func @tensor_dtype_and_device(%tensor: !tvm_ffi_abi.tensor)
    -> (i8, i8, i16, i32, i32) {
  %code, %bits, %lanes = tvm_ffi_abi.tensor.dtype %tensor : !tvm_ffi_abi.tensor
  %device_type, %device_index = tvm_ffi_abi.tensor.device %tensor : !tvm_ffi_abi.tensor
  return %code, %bits, %lanes, %device_type, %device_index
     : i8, i8, i16, i32, i32
}

// -----

// CHECK-LABEL: func.func @tensor_metadata(
// CHECK-SAME: %[[TENSOR0:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[OBJECT0:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR0]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR0:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT0]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[DIM:[a-zA-Z0-9_]+]] = dlpack.tensor.dim %[[DLTENSOR0]] : !dlpack.dl_tensor
// CHECK: %[[OBJECT1:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR0]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR1:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT1]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[SIZE:[a-zA-Z0-9_]+]] = dlpack.tensor.size %[[DLTENSOR1]][%[[INDEX]]] : !dlpack.dl_tensor
// CHECK: %[[OBJECT2:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR0]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR2:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT2]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[STRIDE:[a-zA-Z0-9_]+]] = dlpack.tensor.stride %[[DLTENSOR2]][%[[INDEX]]] : !dlpack.dl_tensor
// CHECK: %[[OBJECT3:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[TENSOR0]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: %[[DLTENSOR3:[a-zA-Z0-9_]+]] = tvm_ffi_abi.as %[[OBJECT3]] : !tvm_ffi_abi.object -> !dlpack.dl_tensor
// CHECK-NEXT: %[[OFFSET:[a-zA-Z0-9_]+]] = dlpack.tensor.storage_offset %[[DLTENSOR3]] : !dlpack.dl_tensor
// CHECK: return %[[DIM]], %[[SIZE]], %[[STRIDE]], %[[OFFSET]] : i64, i64, i64, i64
func.func @tensor_metadata(%tensor: !tvm_ffi_abi.tensor, %index: i64)
    -> (i64, i64, i64, i64) {
  %dim = tvm_ffi_abi.tensor.dim %tensor : !tvm_ffi_abi.tensor
  %size = tvm_ffi_abi.tensor.size %tensor[%index] : !tvm_ffi_abi.tensor
  %stride = tvm_ffi_abi.tensor.stride %tensor[%index] : !tvm_ffi_abi.tensor
  %offset = tvm_ffi_abi.tensor.storage_offset %tensor : !tvm_ffi_abi.tensor
  return %dim, %size, %stride, %offset : i64, i64, i64, i64
}
