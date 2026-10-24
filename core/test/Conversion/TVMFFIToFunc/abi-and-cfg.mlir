// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-func --verify-each | FileCheck %s --implicit-check-not=emit_tvm_ffi_abi

// Forward the borrowed view across CFG blocks. Every exit uses the same output slot and original payload type.
// CHECK-LABEL: func.func @__tvm_ffi_cfg(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[VIEW0:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[ARGS]], %[[UNDEF:[a-zA-Z0-9_]+]][0]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[COUNT]], %[[VIEW0]][1]
// CHECK: %[[PACK:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VIEW]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.size %[[PACK]]
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[SIZE]], %[[ZERO:[a-zA-Z0-9_]+]]
// CHECK: cf.cond_br %[[EMPTY]], ^[[LEFT:bb[0-9]+]](%[[PACK]] : !tvm_ffi.pack.arguments), ^[[RIGHT:bb[0-9]+]](%[[PACK]] : !tvm_ffi.pack.arguments)
// CHECK: ^[[LEFT]](%[[LEFT_ARGS:[a-zA-Z0-9_]+]]: !tvm_ffi.pack.arguments):
// CHECK: %[[LEFT_OK:[a-zA-Z0-9_]+]], %[[LEFT_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.call @target(%[[LEFT_ARGS]])
// CHECK: %[[LEFT_READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[LEFT_OK]]
// CHECK: %[[LEFT_STATUS:[a-zA-Z0-9_]+]] = scf.if %[[LEFT_READY]]
// CHECK: %[[LEFT_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[LEFT_VALUE]] : !tvm_ffi.c.any -> !tvm_ffi.c.tensor
// CHECK: %[[LEFT_ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[LEFT_TYPED]]
// CHECK: %[[LEFT_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[LEFT_ANY]]
// CHECK: llvm.store %[[LEFT_ABI]], %[[OUT]]
// CHECK: return %[[LEFT_STATUS]]
// CHECK: ^[[RIGHT]](%[[RIGHT_ARGS:[a-zA-Z0-9_]+]]: !tvm_ffi.pack.arguments):
// CHECK: %[[RIGHT_OK:[a-zA-Z0-9_]+]], %[[RIGHT_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.call @target(%[[RIGHT_ARGS]])
// CHECK: %[[RIGHT_READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[RIGHT_OK]]
// CHECK: %[[RIGHT_STATUS:[a-zA-Z0-9_]+]] = scf.if %[[RIGHT_READY]]
// CHECK: %[[RIGHT_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[RIGHT_VALUE]] : !tvm_ffi.c.any -> !tvm_ffi.c.tensor
// CHECK: %[[RIGHT_ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RIGHT_TYPED]]
// CHECK: %[[RIGHT_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RIGHT_ANY]]
// CHECK: llvm.store %[[RIGHT_ABI]], %[[OUT]]
// CHECK: return %[[RIGHT_STATUS]]
// CHECK-NOT: func.func @__tvm_ffi_cfg
tvm_ffi.pack.func @cfg(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.c.tensor {
  %size = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
  %zero = arith.constant 0 : i64
  %empty = arith.cmpi eq, %size, %zero : i64
  cf.cond_br %empty, ^left(%args : !tvm_ffi.pack.arguments), ^right(%args : !tvm_ffi.pack.arguments)
^left(%left_args: !tvm_ffi.pack.arguments):
  %left_ok, %left_value = tvm_ffi.pack.call @target(%left_args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  tvm_ffi.return %left_ok, %left_value : !tvm_ffi.bool, !tvm_ffi.c.any
^right(%right_args: !tvm_ffi.pack.arguments):
  %right_ok, %right_value = tvm_ffi.pack.call @target(%right_args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  tvm_ffi.return %right_ok, %right_value : !tvm_ffi.bool, !tvm_ffi.c.any
}

tvm_ffi.pack.func private @target(!tvm_ffi.pack.arguments) -> !tvm_ffi.c.tensor

// -----

// CHECK-LABEL: func.func @__tvm_ffi_packed(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[VIEW0:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[ARGS]], %[[UNDEF:[a-zA-Z0-9_]+]][0]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[COUNT]], %[[VIEW0]][1]
// CHECK: %[[PACK:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VIEW]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.size %[[PACK]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[OK:[a-zA-Z0-9_]+]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: llvm.store %[[VALUE:[a-zA-Z0-9_]+]], %[[OUT]]
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: scf.yield
// CHECK: return %[[STATUS]] : i32
tvm_ffi.pack.func @packed(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.int {
  %size = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
  %boxed = tvm_ffi.to %size : i64 -> !tvm_ffi.int
  %value = tvm_ffi.upcast %boxed : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// The packed result type is checked only on success; failure never writes out.
// CHECK-LABEL: func.func private @__tvm_ffi_status(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.pack.call @target
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[OK]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (i32) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[VALUE]] : !tvm_ffi.c.any -> !tvm_ffi.c.tensor
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TYPED]]
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ANY]]
// CHECK: llvm.store %[[ABI]], %[[OUT]]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: scf.yield %[[ZERO]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.downcast
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: %[[ERROR:[a-zA-Z0-9_-]+]] = arith.constant -1 : i32
// CHECK: scf.yield %[[ERROR]]
// CHECK: return %[[STATUS]]
tvm_ffi.pack.func private @status(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.c.tensor {
  %ok, %value = tvm_ffi.pack.call @target(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}

// CHECK-LABEL: func.func private @__tvm_ffi_target(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
tvm_ffi.pack.func private @target(!tvm_ffi.pack.arguments) -> !tvm_ffi.c.tensor
