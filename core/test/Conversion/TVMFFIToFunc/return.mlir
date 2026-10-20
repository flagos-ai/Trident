// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-func --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='tvm_ffi.'

// CHECK-LABEL: func.func @explicit_status(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi.any, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[FLAG]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (i32) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.tensor
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TYPED]]
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ANY]]
// CHECK: llvm.store %[[ABI]], %[[SLOT]]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: scf.yield %[[ZERO]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.downcast
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: %[[ERROR:[a-zA-Z0-9_-]+]] = arith.constant -1 : i32
// CHECK: scf.yield %[[ERROR]]
// CHECK: return %[[STATUS]]
// LLVM-LABEL: llvm.func @explicit_status(
tvm_ffi.func @explicit_status(%flag: !tvm_ffi.bool, %payload: !tvm_ffi.any) -> !tvm_ffi.tensor {
  tvm_ffi.return %flag, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// CHECK-LABEL: func.func private @external(
// CHECK-SAME: i64, !llvm.ptr) -> i32
tvm_ffi.func @external(i64) -> !tvm_ffi.int

// CHECK-LABEL: func.func @packed_unit(
// CHECK: tvm_ffi.downcast
// CHECK-LABEL: func.func @__tvm_ffi_packed_unit(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = func.call @packed_unit(%[[SLOT]])
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[STATUS]], %[[ZERO]]
// CHECK: scf.if %[[OK]] -> (i32) {
// CHECK: %[[LOADED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
// CHECK: llvm.store %[[LOADED]], %[[OUT]]
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: scf.yield %[[STATUS]]
// LLVM-LABEL: llvm.func @packed_unit(
// LLVM-LABEL: llvm.func @__tvm_ffi_packed_unit(
tvm_ffi.func @packed_unit() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi} {
  %none = tvm_ffi.constant.none
  %any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}
