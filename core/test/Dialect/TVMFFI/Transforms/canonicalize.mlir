//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -canonicalize | FileCheck %s

// FFI calls are effectful: canonicalization must retain an explicit clone.
// CHECK-LABEL: func.func @clone_side_effect(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.function, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[FORMAT:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]], %[[STATUS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[INPUT]], %[[FORMAT]]) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
// CHECK: cf.assert %[[STATUS]], "clone failed"
// CHECK: return %[[CLONE]] : !tvm_ffi.tensor
func.func @clone_side_effect(%handle: !tvm_ffi.function, %input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %format = tvm_ffi.constant.int 0
  %clone, %status = tvm_ffi.FunctionCall %handle(%input, %format) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
  cf.assert %status, "clone failed"
  return %clone : !tvm_ffi.tensor
}
