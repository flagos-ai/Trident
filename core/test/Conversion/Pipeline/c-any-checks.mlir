// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --lower-to-pack --convert-tvm-ffi-to-func --verify-each | FileCheck %s --check-prefix=ABI
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// ABI-LABEL: func.func @__tvm_ffi_finite_boundary(
// ABI-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr
// ABI: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get %[[PACK:[a-zA-Z0-9_]+]][%[[INDEX:[a-zA-Z0-9_]+]]]
// ABI-NOT: tvm_ffi.isa
// ABI-NOT: tvm_ffi.downcast
// ABI: cf.br ^[[BODY:bb[0-9]+]](%[[VALUE]] : !tvm_ffi.c.any)
// ABI: ^[[BODY]](%[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any):
// ABI: scf.if
// ABI-NOT: tvm_ffi.downcast
// ABI: llvm.store %[[ABI_VALUE:[a-zA-Z0-9_]+]], %[[SLOT]]
// LLVM-LABEL: llvm.func @__tvm_ffi_finite_boundary(
// LLVM-NOT: llvm.call @finite_boundary
// LLVM: llvm.return
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
