// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --reconcile-unrealized-casts --verify-each | FileCheck %s

// CHECK-LABEL: func.func @read(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !llvm.struct<(ptr, i32)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[PACK]][1]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = llvm.sext %[[COUNT]]
// CHECK: %[[NONNEG:[a-zA-Z0-9_]+]] = llvm.icmp "sge" %[[INDEX]], %[[ZERO:[a-zA-Z0-9_]+]]
// CHECK: %[[BELOW:[a-zA-Z0-9_]+]] = llvm.icmp "slt" %[[INDEX]], %[[BOUND:[a-zA-Z0-9_]+]]
// CHECK: %[[VALID:[a-zA-Z0-9_]+]] = llvm.and %[[NONNEG]], %[[BELOW]]
// CHECK: cf.assert %[[VALID]], "packed argument index out of bounds"
// CHECK: %[[DATA:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[PACK]][0]
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DATA]][%[[INDEX]]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
func.func @read(%args: !tvm_ffi.pack.arguments, %index: i64) -> (i64, !tvm_ffi.c.any) {
  %size = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
  %value = tvm_ffi.pack.get %args[%index] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.c.any
  return %size, %value : i64, !tvm_ffi.c.any
}
