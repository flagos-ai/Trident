// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-scf-to-cf --convert-tvm-ffi-to-gpu --verify-each | FileCheck %s

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  // CHECK-LABEL: tvm_ffi.func @monadic_launch(
  // CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
  // CHECK: cf.cond_br %[[READY:[a-zA-Z0-9_]+]], ^[[LAUNCH:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
  // CHECK: ^[[LAUNCH]]:
  // CHECK: gpu.launch_func
  // CHECK: tvm_ffi.return %[[SUCCESS_RESULT:[a-zA-Z0-9_]+]]
  // CHECK: ^[[FAILURE]]:
  // CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch"
  // CHECK: %[[WIDENED:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[EXCEPTION]] : !tvm_ffi.exception -> !tvm_ffi.any
  // CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
  // CHECK: %[[REJECTION:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[WIDENED]], %[[TRUE]] : !tvm_ffi.any, i1 to !tvm_ffi.monad.result<!tvm_ffi.any>
  // CHECK: tvm_ffi.return %[[REJECTION]]
  tvm_ffi.func @monadic_launch(%tensor: !tvm_ffi.tensor, %scalar: i64) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
    %one = arith.constant 1 : i64
    tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr>, %scalar : i64 #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>)
    %widened = tvm_ffi.upcast %tensor : !tvm_ffi.tensor -> !tvm_ffi.any
    %result = tvm_ffi.monad.pure %widened : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>

    tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.any>
  }
}
