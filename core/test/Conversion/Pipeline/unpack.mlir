// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -trident-lowering-pipeline | FileCheck %s

// CHECK-LABEL: llvm.func @unpack(
// CHECK-SAME: %[[PAYLOAD:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SUCCESS:[a-zA-Z0-9_]+]]: i1,
// CHECK: llvm.cond_br %[[SUCCESS]],
// CHECK: llvm.store
// CHECK-NOT: tvm_ffi.monad.unpack
// CHECK: llvm.return
tvm_ffi.func @unpack(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.any>
}
