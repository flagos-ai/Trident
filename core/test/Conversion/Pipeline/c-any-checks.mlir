// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --lower-tvm-ffi-abi-to-pack --convert-tvm-ffi-abi-to-func --verify-each | FileCheck %s --check-prefix=ABI
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi_api.' --implicit-check-not='tvm_ffi_abi.'

// ABI-LABEL: func.func @__tvm_ffi_finite_boundary(
// ABI-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr
// ABI: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get %[[PACK:[a-zA-Z0-9_]+]][%[[INDEX:[a-zA-Z0-9_]+]]]
// ABI-NOT: tvm_ffi_abi.isa
// ABI-NOT: tvm_ffi_abi.downcast
// ABI: cf.br ^[[BODY:bb[0-9]+]](%[[VALUE]] : !tvm_ffi_abi.any)
// ABI: ^[[BODY]](%[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any):
// ABI: scf.if
// ABI-NOT: tvm_ffi_abi.downcast
// ABI: llvm.store %[[ABI_VALUE:[a-zA-Z0-9_]+]], %[[SLOT]]
// LLVM-LABEL: llvm.func @__tvm_ffi_finite_boundary(
// LLVM-NOT: llvm.call @finite_boundary
// LLVM: llvm.return
tvm_ffi_api.func @finite_boundary(%value: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> {
  %payload = tvm_ffi_api.upcast %value : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// Finite constraints are currently erased with the semantic type, including
// unused downcasts.
// LLVM-LABEL: llvm.func @unused_finite_downcast(
// LLVM-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// LLVM-NEXT: llvm.return
func.func @unused_finite_downcast(%value: !tvm_ffi_api.any) {
  %checked = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  return
}
