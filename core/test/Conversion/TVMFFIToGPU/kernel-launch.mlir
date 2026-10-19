// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-gpu | FileCheck %s

// Only kernel lowering runs here: semantic ABI values remain available.
// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK: gpu.binary @kernel
// CHECK-LABEL: func.func @test_kernel_launch(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// CHECK: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]] : !tvm_ffi.tensor -> !tvm_ffi.object
// CHECK: %[[DLTENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[OBJECT]] : !tvm_ffi.object -> !dlpack.dl_tensor
// CHECK: %[[DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[DLTENSOR]] : !dlpack.dl_tensor -> !llvm.ptr
// CHECK: %[[NATIVE_SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR]] : !tvm_ffi.int -> i64
// CHECK: scf.if
// CHECK: %[[LAUNCH_OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]]
// CHECK: %[[LAUNCH_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[LAUNCH_OBJECT]]
// CHECK: %[[LAUNCH_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[LAUNCH_TENSOR]]
// CHECK: %[[LAUNCH_SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR]]
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DEVICE_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[ONE]] x i32
// CHECK: llvm.call @aoti_torch_get_current_device_index(%[[DEVICE_SLOT]]) : (!llvm.ptr) -> i32
// CHECK: %[[DEVICE:[a-zA-Z0-9_]+]] = llvm.load %[[DEVICE_SLOT]] : !llvm.ptr -> i32
// CHECK: %[[CUDA:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[STREAM:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIEnvGetStream(%[[CUDA]], %[[DEVICE]]) : (i32, i32) -> !llvm.ptr
// CHECK: gpu.launch_func <%[[STREAM]] : !llvm.ptr> @kernel::@entry
// CHECK: args(%[[LAUNCH_DATA]] : !llvm.ptr, %[[LAUNCH_SCALAR]] : i64,
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes { gpu.container_module } {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func @test_kernel_launch(
    %tensor: !tvm_ffi.tensor,
    %scalar: !tvm_ffi.int) {
    %c32 = llvm.mlir.constant(32 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c128 = llvm.mlir.constant(128 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %shmem = llvm.mlir.constant(16384 : i32) : i32

    %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%c32, %c16, %c1) : i64 threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor : !tvm_ffi.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %scalar : !tvm_ffi.int #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}
