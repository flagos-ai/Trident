// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: not trident-core-opt %s --allow-unregistered-dialect --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer 2>&1 | FileCheck %s

// CHECK: error: failed to legalize operation 'test.container'
func.func @unsupported(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  "test.container"(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> ()
  return
}
