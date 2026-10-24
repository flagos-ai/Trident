// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-to-pack --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.func '
// RUN: trident-core-opt %s --split-input-file --lower-to-pack --convert-tvm-ffi-to-func --verify-each > /dev/null

// CHECK-LABEL: tvm_ffi.pack.func @branch_argument(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi.pack.arguments) -> !tvm_ffi.c.any
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = tvm_ffi.pack.size %[[PACK]]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[ARITY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[COUNT]], %[[ZERO]]
// CHECK: cf.cond_br %[[ARITY]], ^[[UNPACK:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[UNPACK]]:
// CHECK-NOT: tvm_ffi.pack.get
// CHECK: cf.cond_br
// CHECK: ^[[REJECT]]:
// CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[EXCEPTION]]
// CHECK: tvm_ffi.return %[[OK:[a-zA-Z0-9_]+]], %[[ANY]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 7
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[VALUE]] : !tvm_ffi.int)
// CHECK: ^[[NEXT]](%[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[ARG]]
tvm_ffi.func @branch_argument() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 7
  cf.br ^bb1(%value : !tvm_ffi.int)
^bb1(%arg: !tvm_ffi.int):
  %payload = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @cond_branch_arguments(
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 2
// CHECK: cf.cond_br %[[COND:[a-zA-Z0-9_]+]] weights([90, 10]), ^[[YES:bb[0-9]+]](%[[TRUE]] : !tvm_ffi.int), ^[[NO:bb[0-9]+]](%[[FALSE]] : !tvm_ffi.int)
// CHECK: ^[[YES]](%[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[FIRST]]
// CHECK: tvm_ffi.return
// CHECK: ^[[NO]](%[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[SECOND]]
// CHECK: tvm_ffi.return
tvm_ffi.func @cond_branch_arguments() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %cond = arith.constant true
  %true_value = tvm_ffi.constant.int 1
  %false_value = tvm_ffi.constant.int 2
  cf.cond_br %cond, ^true(%true_value : !tvm_ffi.int),
                         ^false(%false_value : !tvm_ffi.int) {branch_weights = array<i32: 90, 10>}
^true(%true_arg: !tvm_ffi.int):
  %true_payload = tvm_ffi.upcast %true_arg : !tvm_ffi.int -> !tvm_ffi.c.any
  %true_ok = tvm_ffi.constant.bool true
  tvm_ffi.return %true_ok, %true_payload : !tvm_ffi.bool, !tvm_ffi.c.any
^false(%false_arg: !tvm_ffi.int):
  %false_payload = tvm_ffi.upcast %false_arg : !tvm_ffi.int -> !tvm_ffi.c.any
  %false_ok = tvm_ffi.constant.bool true
  tvm_ffi.return %false_ok, %false_payload : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @explicit_status(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi.pack.arguments
// CHECK: %[[RAW_FLAG:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get %[[PACK]][%[[ZERO:[a-zA-Z0-9_]+]]]
// CHECK: %[[RAW_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get %[[PACK]][%[[ONE:[a-zA-Z0-9_]+]]]
// CHECK: tvm_ffi.isa %[[RAW_FLAG]] is !tvm_ffi.bool
// CHECK-NOT: tvm_ffi.isa
// CHECK: cf.cond_br %[[MATCHES:[a-zA-Z0-9_]+]], ^[[ACCEPT:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[ACCEPT]]:
// CHECK: %[[TYPED_FLAG:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[RAW_FLAG]] : !tvm_ffi.c.any -> !tvm_ffi.bool
// CHECK-NOT: tvm_ffi.downcast
// CHECK: cf.br ^[[BODY:bb[0-9]+]](%[[TYPED_FLAG]], %[[RAW_PAYLOAD]] : !tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK: ^[[BODY]](%[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any):
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[FLAG]]
// CHECK: %[[CHECKED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi.c.any) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[PAYLOAD]] : !tvm_ffi.c.any -> !tvm_ffi.c.tensor
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[TYPED]]
// CHECK: scf.yield %[[ANY]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.downcast
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: scf.yield %[[FAILED]]
// CHECK: tvm_ffi.return %[[FLAG]], %[[CHECKED]]
tvm_ffi.func @explicit_status(%flag: !tvm_ffi.bool, %payload: !tvm_ffi.c.any) -> !tvm_ffi.c.tensor {
  tvm_ffi.return %flag, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
}

// CHECK-LABEL: tvm_ffi.pack.func private @external(!tvm_ffi.pack.arguments) -> !tvm_ffi.c.any
// Declarations only need their packed signature; no parameter preflight runs.
tvm_ffi.func private @external(i64) -> !tvm_ffi.int

// CHECK-LABEL: tvm_ffi.pack.func nested @external_nested(!tvm_ffi.pack.arguments) -> !tvm_ffi.c.any
tvm_ffi.func nested @external_nested(!tvm_ffi.tensor) -> !tvm_ffi.none

// CHECK-LABEL: tvm_ffi.pack.func @packed_unit(
// CHECK: scf.if
// CHECK: tvm_ffi.downcast %[[VALUE:[a-zA-Z0-9_]+]] : !tvm_ffi.c.any -> !tvm_ffi.none
// CHECK: tvm_ffi.return
tvm_ffi.func @packed_unit() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi} {
  %none = tvm_ffi.constant.none
  %any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @forward_block_use(
// CHECK: cf.br ^[[BODY:bb[0-9]+]]
// CHECK: ^[[BODY]]:
// CHECK-NEXT: cf.br ^[[DEFINITION:bb[0-9]+]]
// CHECK: ^[[CLEANUP:bb[0-9]+]]:
// CHECK: tvm_ffi.ObjectDecRef %[[ARRAY:[a-zA-Z0-9_]+]] : !tvm_ffi.c.array
// CHECK: tvm_ffi.return
// CHECK: ^[[DEFINITION]]:
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[ARRAY]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[ONE:[a-zA-Z0-9_]+]])
// CHECK: cf.br ^[[CLEANUP]]
tvm_ffi.func @forward_block_use() -> !tvm_ffi.none {
  cf.br ^definition
^cleanup:
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.c.array
  %unit_none = tvm_ffi.constant.none
  %payload = tvm_ffi.upcast %unit_none : !tvm_ffi.none -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
^definition:
  %one = tvm_ffi.constant.int 1
  %function = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
  %array = tvm_ffi.FunctionCall %function : !tvm_ffi.c.function(%one) : (!tvm_ffi.int) -> !tvm_ffi.c.array
  cf.br ^cleanup
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @guarded(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi.pack.arguments
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = tvm_ffi.pack.size %[[PACK]]
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = arith.constant 2 : i64
// CHECK: %[[ARITY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[COUNT]], %[[TWO]]
// CHECK-NEXT: cf.cond_br %[[ARITY]], ^[[UNPACK:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[UNPACK]]:
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get %[[PACK]][%[[ZERO:[a-zA-Z0-9_]+]]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.pack.get %[[PACK]][%[[ONE:[a-zA-Z0-9_]+]]]
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = tvm_ffi.isa %[[FIRST]] is !tvm_ffi.int
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi.isa %[[SECOND]] is !tvm_ffi.bool
// CHECK: %[[MATCHES:[a-zA-Z0-9_]+]] = arith.andi %[[PREFIX:[a-zA-Z0-9_]+]], %[[BOOL]]
// CHECK: cf.cond_br %[[MATCHES]], ^[[ACCEPT:bb[0-9]+]], ^[[REJECT]]
// CHECK: ^[[ACCEPT]]:
// CHECK: %[[ARG0:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[FIRST]] : !tvm_ffi.c.any -> !tvm_ffi.int
// CHECK: %[[ARG1:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[SECOND]] : !tvm_ffi.c.any -> !tvm_ffi.bool
// CHECK: cf.br ^[[BODY:bb[0-9]+]](%[[ARG0]], %[[ARG1]] : !tvm_ffi.int, !tvm_ffi.bool)
// CHECK: ^[[REJECT]]:
// CHECK: tvm_ffi.exception "GuardMatch"
// CHECK: ^[[BODY]](%[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi.bool):
// CHECK: tvm_ffi.upcast %[[INPUT]]
tvm_ffi.func @guarded(%arg0: !tvm_ffi.int, %arg1: !tvm_ffi.bool) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %payload = tvm_ffi.upcast %arg0 : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// Functions without the old opt-in marker also become packed entries.
// CHECK-LABEL: tvm_ffi.pack.func public @no_wrapper(
// CHECK-NOT: emit_tvm_ffi_abi
// CHECK: tvm_ffi.constant.int 3
tvm_ffi.func public @no_wrapper() -> !tvm_ffi.int {
  %value = tvm_ffi.constant.int 3
  %payload = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func private @private_entry(
// CHECK: tvm_ffi.return
tvm_ffi.func private @private_entry() -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %value = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @switch_argument(
// CHECK: ^[[BODY:bb[0-9]+]](%[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[DEFAULT:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[CASE:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: %[[SELECTOR:[a-zA-Z0-9_]+]], %[[DEVICE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device %[[TENSOR]]
// CHECK: cf.switch %[[SELECTOR]] : i32, [
// CHECK: default: ^[[MERGE:bb[0-9]+]](%[[DEFAULT]] : !tvm_ffi.int),
// CHECK: 1: ^[[MERGE]](%[[CASE]] : !tvm_ffi.int)
// CHECK: ^[[MERGE]](%[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[VALUE]]
tvm_ffi.func @switch_argument(
    %tensor: !tvm_ffi.c.tensor,
    %default_value: !tvm_ffi.int,
    %case_value: !tvm_ffi.int) -> !tvm_ffi.int {
  %selector, %device_index = tvm_ffi.tensor.device %tensor : !tvm_ffi.c.tensor
  cf.switch %selector : i32, [
    default: ^merge(%default_value : !tvm_ffi.int),
    1: ^merge(%case_value : !tvm_ffi.int)
  ]
^merge(%value: !tvm_ffi.int):
  %payload = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.c.any
}
// -----

// CHECK-LABEL: tvm_ffi.pack.func @typed(
// CHECK: tvm_ffi.pack.get
// CHECK: tvm_ffi.isa %[[ARG:[a-zA-Z0-9_]+]] is !tvm_ffi.int
// CHECK: tvm_ffi.downcast %[[ARG]]
tvm_ffi.func @typed(%arg: !tvm_ffi.int) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.c.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.c.any
}
