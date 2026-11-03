// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-ascend | FileCheck %s

// CHECK-LABEL: llvm.mlir.global internal constant @_kernel_npubin
// CHECK: llvm.mlir.global internal constant @_kernel_name
// CHECK: llvm.mlir.global internal constant @_kernel_types
// CHECK-LABEL: func.func @launch(
// CHECK: llvm.call @aoti_torch_get_current_device_index
// CHECK: llvm.call @TVMFFIEnvGetStream
// CHECK: llvm.call @tridentAscendLaunch
// CHECK: cf.assert
// CHECK-NOT: tvm_ffi.kernel_launch
// CHECK-NOT: tvm_ffi.ascend_kernel

module {
  tvm_ffi.ascend_kernel @_kernel dense<[1, 2, 3, 4]> : tensor<4xi8> {
    argument_types = ["i64"],
    auto_map_blocks = false,
    enable_task_queue = true,
    force_simt_only = false,
    has_ffts = true,
    kernel_name = "entry",
    lock_init_value = 0 : i64,
    lock_num = 0 : i64,
    mix_mode = "aiv",
    physical_block_count = 32 : i64,
    shared_memory_size = 0 : i64,
    use_v2_launch = false,
    workspace_size = 0 : i64
  }

  func.func @launch(%argument: i64) {
    %one = llvm.mlir.constant(1 : i64) : i64
    tvm_ffi.kernel_launch @_kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%argument : i64
          #tvm_ffi.variable_specialization<kind = i64>,
          %one : i64 #tvm_ffi.constant_specialization<value = 1 : i64>)
    func.return
  }
}
