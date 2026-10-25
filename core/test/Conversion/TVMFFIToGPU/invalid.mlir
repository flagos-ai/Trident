// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-kernel-launch --convert-tvm-ffi-to-gpu -verify-diagnostics

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]
  func.func @invalid_operand(%value: f64) {
    %one = arith.constant 1 : i64
    // expected-error@+2 {{failed to materialize a native kernel operand}}
    // expected-error@+1 {{failed to legalize operation 'tvm_ffi.kernel_launch' that was explicitly marked illegal}}
    %launch = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
        args (%value : f64 #tvm_ffi.variable_specialization<kind = !llvm.ptr>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}
