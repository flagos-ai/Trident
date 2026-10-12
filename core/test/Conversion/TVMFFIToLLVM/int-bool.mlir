//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @int_bool(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: ConvertUnrealizedIntCastOp
// CHECK-NOT: tvm_ffi.to
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = llvm.trunc %[[PAYLOAD]] : i64 to i1
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL]] : i1 to i64
// CHECK: %[[UNDEF_RESULT:[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[TYPE_INDEX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[TYPE_RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[TYPE_INDEX]], %[[UNDEF_RESULT]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[KIND_INDEX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[KIND_RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND_INDEX]], %[[TYPE_RESULT]][1] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[INT]], %[[KIND_RESULT]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @int_bool(%arg0: !tvm_ffi.bool) -> !tvm_ffi.int {
  %bool = tvm_ffi.get %arg0 : !tvm_ffi.bool -> i1
  %int = arith.extui %bool : i1 to i64
  %0 = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  return %0 : !tvm_ffi.int
}
