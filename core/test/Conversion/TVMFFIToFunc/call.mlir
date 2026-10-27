// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-func --verify-each | FileCheck %s

func.func private @callee(!tvm_ffi.c.tensor, !tvm_ffi.int, !llvm.ptr) -> i32
func.func private @empty(!llvm.ptr) -> i32
func.func private @__tvm_ffi_callee(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
func.func private @__tvm_ffi_empty(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.c.tensor -> !tvm_ffi.c.any
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[BOXED]]
// CHECK: %[[FIRST_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK: llvm.store %[[ABI]], %[[FIRST_SLOT]]
// CHECK: %[[SECOND_BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[SECOND]] : !tvm_ffi.int -> !tvm_ffi.c.any
// CHECK: %[[SECOND_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[SECOND_BOXED]]
// CHECK: %[[SECOND_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[SECOND_ABI]], %[[SECOND_SLOT]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = call @__tvm_ffi_callee(%[[CTX:[a-zA-Z0-9_]+]], %[[ARGS]], %[[COUNT:[a-zA-Z0-9_]+]], %[[OUTPUT]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS]], %[[ZERO:[a-zA-Z0-9_]+]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = scf.if %[[OK]] -> (!tvm_ffi.c.any) {
// CHECK-NEXT: %[[LOADED:[a-zA-Z0-9_]+]] = llvm.load %[[OUTPUT]]
// CHECK-NEXT: %[[PAYLOAD:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[LOADED]]
// CHECK-NEXT: scf.yield %[[PAYLOAD]]
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK: tvm_ffi.constant.none
// CHECK: scf.yield
// CHECK: %[[FLAG:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[OK]]
// CHECK: return %[[FLAG]], %[[VALUE]]
func.func @invoke(%input: !tvm_ffi.c.tensor, %second: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any) {
  %ok, %value = tvm_ffi.call @callee(%input, %second) : (!tvm_ffi.c.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}
// CHECK-LABEL: func.func @invoke_empty(
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64)
// CHECK: llvm.alloca %[[COUNT]]
// CHECK: call @__tvm_ffi_empty
// CHECK: scf.if
func.func @invoke_empty() -> (!tvm_ffi.bool, !tvm_ffi.c.any) {
  %ok, %value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}

// Repeated calls reuse the wrapper supplied by TVMFFIToFunc.
func.func private @repeated(!llvm.ptr) -> i32
func.func private @__tvm_ffi_repeated(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @invoke_repeated(
// CHECK: call @__tvm_ffi_repeated
// CHECK: call @__tvm_ffi_repeated
func.func @invoke_repeated() {
  %first_ok, %first = tvm_ffi.call @repeated() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  %second_ok, %second = tvm_ffi.call @repeated() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return
}
