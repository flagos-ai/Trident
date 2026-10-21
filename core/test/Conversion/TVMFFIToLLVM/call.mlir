// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.call' --implicit-check-not='scf.' --implicit-check-not='ffi.Array' --implicit-check-not='TVMFFIFunction' --implicit-check-not='TVMFFIObject'

func.func private @callee(!tvm_ffi.c.tensor, !tvm_ffi.int, !llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi}
func.func private @empty(!llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi}
func.func private @__tvm_ffi_callee(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
func.func private @__tvm_ffi_empty(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SECOND:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64)
// CHECK-NEXT: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[TWO]]
// CHECK-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK-NEXT: llvm.store %[[INPUT]], %[[FIRST]]
// CHECK-NEXT: %[[SECOND_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK-NEXT: llvm.store %[[SECOND]], %[[SECOND_SLOT]]
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK-NEXT: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store %[[ZERO]], %[[OUTPUT]]
// CHECK-NEXT: %[[CONTEXT:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32)
// CHECK-NEXT: %[[STATUS:[a-zA-Z0-9_]+]] = call @__tvm_ffi_callee(%[[CONTEXT]], %[[ARGS]], %[[COUNT]], %[[OUTPUT]])
// CHECK-NEXT: %[[SUCCESS_STATUS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS]], %[[SUCCESS_STATUS]]
// CHECK-NEXT: %[[FLAG_BITS:[a-zA-Z0-9_]+]] = llvm.zext %[[OK]] : i1 to i64
// CHECK: %[[FLAG:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[FLAG_BITS]], %[[FLAG_PREFIX:[a-zA-Z0-9_]+]][2]
// CHECK-NEXT: %[[NONE:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.cond_br %[[OK]], ^[[PASSED:bb[0-9]+]], ^[[CONTINUE:bb[0-9]+]](%[[NONE]] :
// CHECK: ^[[PASSED]]:
// CHECK-NEXT: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.load %[[OUTPUT]]
// CHECK-NEXT: llvm.br ^[[CONTINUE]](%[[PAYLOAD]] :
// CHECK: ^[[CONTINUE]](%[[VALUE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>):
// CHECK-NEXT: return %[[FLAG]], %[[VALUE]]
func.func @invoke(%input: !tvm_ffi.c.tensor, %second: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any) {
  %ok, %value = tvm_ffi.call @callee(%input, %second) : (!tvm_ffi.c.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}
// CHECK-LABEL: func.func @invoke_empty(
// CHECK: %[[EMPTY_COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64)
// CHECK-NEXT: %[[EMPTY_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[EMPTY_COUNT]]
// CHECK: %[[EMPTY_OUTPUT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[EMPTY_CONTEXT:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[EMPTY_ARITY:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32)
// CHECK-NEXT: %[[EMPTY_STATUS:[a-zA-Z0-9_]+]] = call @__tvm_ffi_empty(%[[EMPTY_CONTEXT]], %[[EMPTY_ARGS]], %[[EMPTY_ARITY]], %[[EMPTY_OUTPUT]])
// CHECK: llvm.icmp "eq" %[[EMPTY_STATUS]], %[[EMPTY_ZERO:[a-zA-Z0-9_]+]]
func.func @invoke_empty() -> (!tvm_ffi.bool, !tvm_ffi.c.any) {
  %ok, %value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}

// Repeated calls reuse the wrapper supplied by TVMFFIToFunc.
func.func private @repeated(!llvm.ptr) -> i32 attributes {emit_tvm_ffi_abi}
func.func private @__tvm_ffi_repeated(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-LABEL: func.func @invoke_repeated(
// CHECK: call @__tvm_ffi_repeated
// CHECK: call @__tvm_ffi_repeated
func.func @invoke_repeated() {
  %first_ok, %first = tvm_ffi.call @repeated() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  %second_ok, %second = tvm_ffi.call @repeated() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
  return
}
