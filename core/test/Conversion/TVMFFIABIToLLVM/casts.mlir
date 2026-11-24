// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -canonicalize --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer -convert-tvm-ffi-abi-to-llvm -canonicalize | FileCheck %s

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
func.func @checked_int(%value: !tvm_ffi_api.any) -> !tvm_ffi_abi.int {
  %result = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// Finite Union constraints are deferred; exporting the CAny value still retains it.
// CHECK-LABEL: func.func @erased_union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-NOT: llvm.call @abort()
// CHECK: llvm.call @TVMFFIObjectIncRef
// CHECK-NOT: llvm.call @abort()
// CHECK: return %[[VALUE]]
func.func @erased_union(%value: !tvm_ffi_api.any) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> {
  %result = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
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
func.func @unused_checks(%first: !tvm_ffi_api.any, %second: !tvm_ffi_api.union<!tvm_ffi_abi.tensor, !tvm_ffi_abi.none>) {
  %a = tvm_ffi_api.downcast %first : !tvm_ffi_api.any -> !tvm_ffi_abi.array
  %b = tvm_ffi_api.downcast %second : !tvm_ffi_api.union<!tvm_ffi_abi.tensor, !tvm_ffi_abi.none> -> !tvm_ffi_abi.tensor
  return
}

// CHECK-LABEL: func.func @string_cast(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK-NOT: llvm.call @abort()
// CHECK: return %[[INPUT]]
func.func @string_cast(%value: !tvm_ffi_api.any) -> !tvm_ffi_api.str {
  %string = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_api.str
  return %string : !tvm_ffi_api.str
}
