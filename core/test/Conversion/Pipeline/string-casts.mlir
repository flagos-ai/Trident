// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s

// CHECK-LABEL: func.func @string_cast(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK-NOT: llvm.call @abort()
// CHECK: return %[[INPUT]]
func.func @string_cast(%value: !tvm_ffi.any) -> !tvm_ffi.str {
  %string = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.str
  return %string : !tvm_ffi.str
}
