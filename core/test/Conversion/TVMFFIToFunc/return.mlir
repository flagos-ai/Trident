// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --convert-tvm-ffi-to-func --verify-each | FileCheck %s
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.'

// LLVM-LABEL: llvm.func @branches(
// CHECK-LABEL: func.func @branches(
// CHECK-SAME: %[[CONDITION:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: cf.cond_br %[[CONDITION]], ^[[YES:bb[0-9]+]](%[[INPUT]] : !tvm_ffi.monad.result<!tvm_ffi.int>), ^[[NO:bb[0-9]+]](%[[INPUT]] : !tvm_ffi.monad.result<!tvm_ffi.int>)
// CHECK: ^[[YES]](%[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>):
// CHECK: %[[FIRST_READY:[a-zA-Z0-9_]+]], %[[FIRST_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[FIRST]]
// CHECK: scf.if %[[FIRST_READY]] -> (i32) {
// CHECK: %[[FIRST_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[FIRST_VALUE]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[FIRST_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[FIRST_TYPED]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[FIRST_ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: arith.constant -1 : i32
// CHECK: return %[[FIRST_STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK: ^[[NO]](%[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>):
// CHECK: %[[SECOND_READY:[a-zA-Z0-9_]+]], %[[SECOND_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[SECOND]]
// CHECK: scf.if %[[SECOND_READY]] -> (i32)
// CHECK: %[[SECOND_TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[SECOND_VALUE]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[SECOND_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[SECOND_TYPED]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[SECOND_ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: return %[[SECOND_STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @branches(%condition: i1, %input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  cf.cond_br %condition, ^yes(%input : !tvm_ffi.monad.result<!tvm_ffi.int>), ^no(%input : !tvm_ffi.monad.result<!tvm_ffi.int>)
^yes(%first: !tvm_ffi.monad.result<!tvm_ffi.int>):
  tvm_ffi.return %first : !tvm_ffi.monad.result<!tvm_ffi.int>
^no(%second: !tvm_ffi.monad.result<!tvm_ffi.int>):
  tvm_ffi.return %second : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// CHECK-LABEL: func.func private @external(
// CHECK-SAME: i64, !llvm.ptr) -> i32
tvm_ffi.func @external(i64) -> !tvm_ffi.monad.result<!tvm_ffi.int>

// LLVM-LABEL: llvm.func @handle(
// CHECK-LABEL: func.func @handle(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.function>, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: scf.if %[[OK]]
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[PAYLOAD]] : !tvm_ffi.function to !llvm.ptr
// CHECK: llvm.store %[[NATIVE]], %[[SLOT]] : !llvm.ptr, !llvm.ptr
tvm_ffi.func @handle(%input: !tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.function> {
  tvm_ffi.return %input : !tvm_ffi.monad.result<!tvm_ffi.function>
}

// CHECK-LABEL: func.func @ordinary_boolean(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[VALUE]]
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[PURE]]
// CHECK: scf.if %[[OK]] -> (i32)
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.bool
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[BOOL]] : !tvm_ffi.bool to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @ordinary_boolean(%value: !tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.bool> {
  %returned = tvm_ffi.monad.pure %value {tvm_ffi.cleanup} : (!tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.bool>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.bool>
}

// CHECK-LABEL: func.func @packed_unit(
// CHECK-SAME: %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]]
// CHECK: %[[READY:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[PURE]]
// CHECK: scf.if %[[READY]] -> (i32) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: %[[NONE_ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[TYPED]] : !tvm_ffi.none to !llvm.struct<(i32, i32, i64)>
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
  %unit = tvm_ffi.monad.pure %unit_none {tvm_ffi.cleanup} : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: func.func @unit(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.none>, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: scf.if %[[SUCCESS]] -> (i32)
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.any -> !tvm_ffi.none
// CHECK: %[[ABI:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]] : !tvm_ffi.none to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ABI]], %[[SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: } else {
// CHECK-NOT: llvm.store
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @unit(%input: !tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  tvm_ffi.return %input : !tvm_ffi.monad.result<!tvm_ffi.none>
}
