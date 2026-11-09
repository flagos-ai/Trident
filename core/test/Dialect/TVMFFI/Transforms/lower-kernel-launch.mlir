// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-kernel-launch --verify-each | FileCheck %s --implicit-check-not='specialization<' --implicit-check-not='gpu.launch_func' --implicit-check-not='tvm_ffi.kernel_launch'

// CHECK-LABEL: func.func @monadic_launch(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SCALAR:[a-zA-Z0-9_]+]]: i64
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[READY:[a-zA-Z0-9_]+]] -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
// CHECK: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[OBJECT]]
// CHECK: %[[DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[VIEW]]
// CHECK: tvm_ffi.native_kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[DATA]], %[[SCALAR]] : !llvm.ptr, i64)
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]]
// CHECK: scf.yield %[[PURE]]
// CHECK: } else {
// CHECK: %[[FAIL:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[FAIL]]
// CHECK: return %[[BRANCH]]

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func @monadic_launch(%tensor: !tvm_ffi.c.tensor, %scalar: i64, %constant: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr>, %scalar : i64 #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>, %constant : !tvm_ffi.int #tvm_ffi.constant_specialization<value = 7 : i64>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return %launch : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}

