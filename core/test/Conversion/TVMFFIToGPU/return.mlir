// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-gpu --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.exception' --implicit-check-not='tvm_ffi.return'

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  // CHECK-LABEL: func.func @monadic_launch(
  // CHECK: %[[OUTCOME:[a-zA-Z0-9_]+]] = scf.if %[[READY:[a-zA-Z0-9_]+]] -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
  // CHECK: gpu.launch_func
  // CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
  // CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]]
  // CHECK: scf.yield %[[SUCCESS]]
  // CHECK: } else {
  // CHECK: %[[REJECTION:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
  // CHECK: scf.yield %[[REJECTION]]
  // CHECK: return %[[OUTCOME]]
  func.func @monadic_launch(%tensor: !tvm_ffi.tensor, %scalar: i64) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr>, %scalar : i64 #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return %launch : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}
