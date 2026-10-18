// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-tvm-ffi-to-func --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-tvm-ffi-to-func --canonicalize --verify-each | FileCheck %s --check-prefix=CLEAN --implicit-check-not='!tvm_ffi.monad.result' --implicit-check-not='tvm_ffi.return'

// CHECK-LABEL: func.func @branches(
// CHECK-SAME: %[[CONDITION:[a-zA-Z0-9_]+]]: i1, %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[READY:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: cf.cond_br %[[CONDITION]], ^[[YES:bb[0-9]+]](%[[VALUE]], %[[READY]] : i64, i1), ^[[NO:bb[0-9]+]](%[[VALUE]], %[[READY]] : i64, i1)
// CHECK: ^[[YES]](%[[FIRST:[a-zA-Z0-9_]+]]: i64, %[[FIRST_READY:[a-zA-Z0-9_]+]]: i1):
// CHECK: scf.if %[[FIRST_READY]] -> (i32)
// CHECK: llvm.store %[[FIRST]], %[[SLOT]] : i64, !llvm.ptr
// CHECK: return %[[FIRST_STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK: ^[[NO]](%[[SECOND:[a-zA-Z0-9_]+]]: i64, %[[SECOND_READY:[a-zA-Z0-9_]+]]: i1):
// CHECK: scf.if %[[SECOND_READY]] -> (i32)
// CHECK: llvm.store %[[SECOND]], %[[SLOT]] : i64, !llvm.ptr
// CHECK: return %[[SECOND_STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @branches(%condition: i1, %input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<i64> {
  cf.cond_br %condition, ^yes(%input : !tvm_ffi.monad.result<i64>), ^no(%input : !tvm_ffi.monad.result<i64>)
^yes(%first: !tvm_ffi.monad.result<i64>):
  tvm_ffi.return %first : !tvm_ffi.monad.result<i64>
^no(%second: !tvm_ffi.monad.result<i64>):
  tvm_ffi.return %second : !tvm_ffi.monad.result<i64>
}

// CHECK-LABEL: func.func private @external(
// CHECK-SAME: i64, !llvm.ptr, !llvm.ptr) -> i32
tvm_ffi.func @external(i64) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>)

// CHECK-LABEL: func.func @mixed(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[PAYLOAD_STATUS:[a-zA-Z0-9_]+]]: i1, %[[NONE:[a-zA-Z0-9_]+]]: !tvm_ffi.none, %[[UNIT_STATUS:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NONE_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = arith.andi %[[PAYLOAD_STATUS]], %[[UNIT_STATUS]] : i1
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (i32) {
// CHECK: llvm.store %[[VALUE]], %[[SLOT]] : i64, !llvm.ptr
// CHECK: %[[NONE_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]] : !tvm_ffi.none to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[NONE_ABI]], %[[NONE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: scf.yield %[[ZERO]] : i32
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: %[[ERROR:[a-zA-Z0-9_-]+]] = arith.constant -1 : i32
// CHECK: scf.yield %[[ERROR]] : i32
// CHECK: return %[[STATUS]] : i32
// CHECK-NOT: tvm_ffi.return
// CHECK-NOT: !tvm_ffi.monad.result
tvm_ffi.func @mixed(%input: !tvm_ffi.monad.result<i64>, %unit: !tvm_ffi.monad.result<!tvm_ffi.none>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  tvm_ffi.return %input, %unit : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: func.func @native_index(
// CLEAN-LABEL: func.func @native_index(
// CLEAN-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: index, %[[READY:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CLEAN: scf.if %[[READY]] {
// CLEAN: %[[NATIVE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VALUE]] : index to i64
// CLEAN: llvm.store %[[NATIVE]], %[[SLOT]] : i64, !llvm.ptr
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: index, %[[READY:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: scf.if %[[READY]] -> (i32)
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[VALUE]] : index to i64
// CHECK: llvm.store %[[NATIVE]], %[[SLOT]] : i64, !llvm.ptr
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @native_index(%input: !tvm_ffi.monad.result<index>) -> !tvm_ffi.monad.result<index> {
  tvm_ffi.return %input : !tvm_ffi.monad.result<index>
}

// CHECK-LABEL: func.func @ordinary_boolean(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: llvm.store %[[VALUE]], %[[SLOT]] : i1, !llvm.ptr
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @ordinary_boolean(%value: i1) -> !tvm_ffi.monad.result<i1> {
  %returned = tvm_ffi.monad.pure %value : (i1) -> !tvm_ffi.monad.result<i1>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<i1>
}

// CHECK-LABEL: func.func @packed_unit(
// CHECK-SAME: %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: scf.if %[[READY]] -> (i32) {
// CHECK: %[[NONE_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]] : !tvm_ffi.none to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[NONE_ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: return %[[UNIT_STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: func.func @__tvm_ffi_packed_unit(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[CALL_STATUS:[a-zA-Z0-9_]+]] = func.call @packed_unit(%[[SLOT]]) : (!llvm.ptr) -> i32
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[CALL_STATUS]], %[[ZERO]] : i32
// CHECK: %[[PACKED_STATUS:[a-zA-Z0-9_]+]] = scf.if %[[OK]] -> (i32) {
// CHECK: %[[LOADED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.store %[[LOADED]], %[[OUT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: } else {
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK: scf.yield %[[CALL_STATUS]] : i32
tvm_ffi.func @packed_unit() -> !tvm_ffi.monad.result<!tvm_ffi.none> attributes {emit_tvm_ffi_abi} {
  %unit_none = tvm_ffi.constant.none
  %unit = tvm_ffi.monad.pure %unit_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: func.func @repeated(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: i64, %[[SUCCESS:[a-zA-Z0-9_]+]]: i1, %[[FIRST:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[SECOND:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: llvm.store %[[VALUE]], %[[FIRST]] : i64, !llvm.ptr
// CHECK: llvm.store %[[VALUE]], %[[SECOND]] : i64, !llvm.ptr
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @repeated(%input: !tvm_ffi.monad.result<i64>) -> (!tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i64>) {
  tvm_ffi.return %input, %input : !tvm_ffi.monad.result<i64>, !tvm_ffi.monad.result<i64>
}

// CHECK-LABEL: func.func @unit(
// CHECK-SAME: %[[NONE:[a-zA-Z0-9_]+]]: !tvm_ffi.none, %[[SUCCESS:[a-zA-Z0-9_]+]]: i1, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: scf.if %[[SUCCESS]] -> (i32)
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]] : !tvm_ffi.none to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @unit(%input: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  tvm_ffi.return %input : !tvm_ffi.monad.result<!tvm_ffi.none>
}
