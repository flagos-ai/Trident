// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -canonicalize --decompose-tvm-ffi --lower-tvm-ffi-defer -convert-tvm-ffi-to-llvm -canonicalize | FileCheck %s

// CHECK: llvm.func @abort()
// CHECK-LABEL: func.func @checked_int(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK: %[[EXPECTED:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(false) : i1
// CHECK: %[[TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[VALUE]][0]
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[TAG]], %[[EXPECTED]]
// CHECK: %[[CONDITION:[a-zA-Z0-9_]+]] = llvm.or %[[FALSE]], %[[MATCH]]
// CHECK: llvm.cond_br %[[CONDITION]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[FAILURE]]:
// CHECK: llvm.call @abort()
// CHECK-NEXT: llvm.unreachable
// CHECK: ^[[SUCCESS]]:
// CHECK: return %[[VALUE]]
func.func @checked_int(%value: !tvm_ffi.any) -> !tvm_ffi.int {
  %result = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// Finite Union constraints are deferred; exporting the CAny value still retains it.
// CHECK-LABEL: func.func @erased_union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-NOT: llvm.call @abort()
// CHECK: llvm.call @TVMFFIObjectIncRef
// CHECK-NOT: llvm.call @abort()
// CHECK: return %[[VALUE]]
func.func @erased_union(%value: !tvm_ffi.any) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
}

// Both unused checks must survive canonicalization.
// CHECK-LABEL: func.func @unused_checks(
// CHECK: llvm.cond_br
// CHECK: llvm.call @abort()
// CHECK: llvm.unreachable
// CHECK: llvm.cond_br
// CHECK: llvm.call @abort()
// CHECK: llvm.unreachable
// CHECK: return
func.func @unused_checks(%first: !tvm_ffi.any, %second: !tvm_ffi.union<!tvm_ffi.c.tensor, !tvm_ffi.none>) {
  %a = tvm_ffi.downcast %first : !tvm_ffi.any -> !tvm_ffi.c.array
  %b = tvm_ffi.downcast %second : !tvm_ffi.union<!tvm_ffi.c.tensor, !tvm_ffi.none> -> !tvm_ffi.c.tensor
  return
}

// CHECK-LABEL: func.func @string_cast(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK-NOT: llvm.call @abort()
// CHECK: return %[[INPUT]]
func.func @string_cast(%value: !tvm_ffi.any) -> !tvm_ffi.str {
  %string = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.str
  return %string : !tvm_ffi.str
}
