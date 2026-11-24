// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-abi-to-func --verify-each | FileCheck %s

// Forward the borrowed view across CFG blocks. Every exit uses the same output slot and original payload type.
// CHECK-LABEL: func.func @__tvm_ffi_cfg(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[VIEW0:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[ARGS]], %[[UNDEF:[a-zA-Z0-9_]+]][0]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[COUNT]], %[[VIEW0]][1]
// CHECK: %[[PACK:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VIEW]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.size %[[PACK]]
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[SIZE]], %[[ZERO:[a-zA-Z0-9_]+]]
// CHECK: cf.cond_br %[[EMPTY]], ^[[LEFT:bb[0-9]+]](%[[PACK]] : !tvm_ffi_abi.pack.arguments), ^[[RIGHT:bb[0-9]+]](%[[PACK]] : !tvm_ffi_abi.pack.arguments)
// CHECK: ^[[LEFT]](%[[LEFT_ARGS:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.pack.arguments):
// CHECK: %[[LEFT_CALL:[a-zA-Z0-9_]+]] = call @__tvm_ffi_target
// CHECK: %[[LEFT_SUCCESS:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[LEFT_CALL]], %[[LEFT_ZERO_STATUS:[a-zA-Z0-9_]+]]
// CHECK: %[[LEFT_VALUE:[a-zA-Z0-9_]+]] = scf.if %[[LEFT_SUCCESS]]
// CHECK: llvm.load
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK: tvm_ffi_abi.constant.none
// CHECK: %[[LEFT_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[LEFT_SUCCESS]]
// CHECK: %[[LEFT_READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[LEFT_OK]]
// CHECK: %[[LEFT_STATUS:[a-zA-Z0-9_]+]] = scf.if %[[LEFT_READY]]
// CHECK: %[[LEFT_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[LEFT_VALUE]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.tensor
// CHECK: %[[LEFT_ANY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[LEFT_TYPED]]
// CHECK: %[[LEFT_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[LEFT_ANY]]
// CHECK: llvm.store %[[LEFT_ABI]], %[[OUT]]
// CHECK: return %[[LEFT_STATUS]]
// CHECK: ^[[RIGHT]](%[[RIGHT_ARGS:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.pack.arguments):
// CHECK: %[[RIGHT_CALL:[a-zA-Z0-9_]+]] = call @__tvm_ffi_target
// CHECK: %[[RIGHT_SUCCESS:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[RIGHT_CALL]], %[[RIGHT_ZERO_STATUS:[a-zA-Z0-9_]+]]
// CHECK: %[[RIGHT_VALUE:[a-zA-Z0-9_]+]] = scf.if %[[RIGHT_SUCCESS]]
// CHECK: llvm.load
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK: tvm_ffi_abi.constant.none
// CHECK: %[[RIGHT_OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[RIGHT_SUCCESS]]
// CHECK: %[[RIGHT_READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[RIGHT_OK]]
// CHECK: %[[RIGHT_STATUS:[a-zA-Z0-9_]+]] = scf.if %[[RIGHT_READY]]
// CHECK: %[[RIGHT_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[RIGHT_VALUE]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.tensor
// CHECK: %[[RIGHT_ANY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[RIGHT_TYPED]]
// CHECK: %[[RIGHT_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[RIGHT_ANY]]
// CHECK: llvm.store %[[RIGHT_ABI]], %[[OUT]]
// CHECK: return %[[RIGHT_STATUS]]
// CHECK-NOT: func.func @__tvm_ffi_cfg
tvm_ffi_abi.pack.func @cfg(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.tensor {
  %size = tvm_ffi_abi.pack.size %args : !tvm_ffi_abi.pack.arguments
  %zero = arith.constant 0 : i64
  %empty = arith.cmpi eq, %size, %zero : i64
  cf.cond_br %empty, ^left(%args : !tvm_ffi_abi.pack.arguments), ^right(%args : !tvm_ffi_abi.pack.arguments)
^left(%left_args: !tvm_ffi_abi.pack.arguments):
  %left_ok, %left_value = tvm_ffi_abi.pack.call @target(%left_args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
  tvm_ffi_abi.return %left_ok, %left_value : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
^right(%right_args: !tvm_ffi_abi.pack.arguments):
  %right_ok, %right_value = tvm_ffi_abi.pack.call @target(%right_args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
  tvm_ffi_abi.return %right_ok, %right_value : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}

tvm_ffi_abi.pack.func private @target(!tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.tensor

// -----

// CHECK-LABEL: func.func @__tvm_ffi_packed(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[VIEW0:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[ARGS]], %[[UNDEF:[a-zA-Z0-9_]+]][0]
// CHECK: %[[VIEW:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[COUNT]], %[[VIEW0]][1]
// CHECK: %[[PACK:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VIEW]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.size %[[PACK]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[OK:[a-zA-Z0-9_]+]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]]
// CHECK: llvm.store %[[VALUE:[a-zA-Z0-9_]+]], %[[OUT]]
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: scf.yield
// CHECK: return %[[STATUS]] : i32
tvm_ffi_abi.pack.func @packed(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.int {
  %size = tvm_ffi_abi.pack.size %args : !tvm_ffi_abi.pack.arguments
  %boxed = tvm_ffi_abi.to %size : i64 -> !tvm_ffi_abi.int
  %value = tvm_ffi_abi.upcast %boxed : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}

// -----

// The packed result type is checked only on success; failure never writes out.
// CHECK-LABEL: func.func private @__tvm_ffi_status(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = call @__tvm_ffi_target
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[CALL]], %[[ZERO_STATUS:[a-zA-Z0-9_]+]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = scf.if %[[SUCCESS]]
// CHECK: llvm.load
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK: tvm_ffi_abi.constant.none
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[SUCCESS]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[OK]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (i32) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[VALUE]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.tensor
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[TYPED]]
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ANY]]
// CHECK: llvm.store %[[ABI]], %[[OUT]]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: scf.yield %[[ZERO]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi_abi.downcast
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: %[[ERROR:[a-zA-Z0-9_-]+]] = arith.constant -1 : i32
// CHECK: scf.yield %[[ERROR]]
// CHECK: return %[[STATUS]]
tvm_ffi_abi.pack.func private @status(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.tensor {
  %ok, %value = tvm_ffi_abi.pack.call @target(%args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
  tvm_ffi_abi.return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}

// CHECK-LABEL: func.func private @__tvm_ffi_target(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
tvm_ffi_abi.pack.func private @target(!tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.tensor
