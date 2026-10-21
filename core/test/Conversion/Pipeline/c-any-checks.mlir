// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --convert-tvm-ffi-to-func --verify-each | FileCheck %s --check-prefix=ABI
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// ABI-LABEL: func.func @finite_boundary(
// ABI-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr
// ABI: scf.if %[[SUCCESS:[a-zA-Z0-9_]+]] -> (i32) {
// ABI: llvm.store %[[ABI_VALUE:[a-zA-Z0-9_]+]], %[[SLOT]]
// ABI: } else {
// ABI-NOT: llvm.store
// ABI: scf.yield
// ABI-LABEL: func.func @__tvm_ffi_finite_boundary(
// ABI: %[[LOADED:[a-zA-Z0-9_]+]] = llvm.load %[[ARG_SLOT:[a-zA-Z0-9_]+]]
// ABI-NOT: llvm.extractvalue
// ABI-NOT: llvm.icmp
// ABI: scf.if
// ABI: func.call @finite_boundary
// LLVM-LABEL: llvm.func @finite_boundary(
// LLVM-NOT: llvm.call @abort()
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_finite_boundary(
// LLVM: llvm.call @finite_boundary
tvm_ffi.func @finite_boundary(%value: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> attributes {emit_tvm_ffi_abi} {
  %payload = tvm_ffi.upcast %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// Finite constraints are currently erased with the semantic type, including
// unused downcasts.
// LLVM-LABEL: llvm.func @unused_finite_downcast(
// LLVM-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// LLVM-NEXT: llvm.return
func.func @unused_finite_downcast(%value: !tvm_ffi.any) {
  %checked = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return
}
