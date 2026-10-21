// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-defer --split-input-file --verify-diagnostics

func.func @loop() {
  %lower = arith.constant 0 : index
  %upper = arith.constant 1 : index
  scf.for %i = %lower to %upper step %upper {
    // expected-error@+1 {{unsupported enclosing operation for deferred cleanup}}
    tvm_ffi.defer {
      tvm_ffi.defer_yield
    }
  }
  return
}

// -----

func.func @multiblock() {
  // expected-error@+1 {{requires a single-block enclosing region}}
  tvm_ffi.defer {
    tvm_ffi.defer_yield
  }
  cf.br ^exit
^exit:
  return
}

// -----

func.func @unreachable_exit() {
  // expected-error@+1 {{unsupported exit for deferred cleanup}}
  tvm_ffi.defer {
    tvm_ffi.defer_yield
  }
  llvm.unreachable
}
