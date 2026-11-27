// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-abi-to-pack --verify-each | FileCheck %s --implicit-check-not='tvm_ffi_abi.func{{[[:space:]]}}'
// RUN: trident-core-opt %s --split-input-file --lower-tvm-ffi-abi-to-pack --convert-tvm-ffi-abi-to-func --verify-each > /dev/null

// CHECK-LABEL: tvm_ffi_abi.pack.func @branch_argument(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.any
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.size %[[PACK]]
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[ARITY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[COUNT]], %[[ZERO]]
// CHECK: cf.cond_br %[[ARITY]], ^[[UNPACK:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[UNPACK]]:
// CHECK-NOT: tvm_ffi_abi.pack.get
// CHECK: cf.cond_br
// CHECK: ^[[REJECT]]:
// CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi_abi.exception "GuardMatch" : !tvm_ffi_abi.exception
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[EXCEPTION]]
// CHECK: tvm_ffi_abi.return %[[OK:[a-zA-Z0-9_]+]], %[[ANY]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.int 7
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[VALUE]] : !tvm_ffi_abi.int)
// CHECK: ^[[NEXT]](%[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK: tvm_ffi_abi.upcast %[[ARG]]
tvm_ffi_abi.func @branch_argument() -> !tvm_ffi_abi.int {
  %value = tvm_ffi_abi.constant.int 7
  cf.br ^bb1(%value : !tvm_ffi_abi.int)
^bb1(%arg: !tvm_ffi_abi.int):
  %payload = tvm_ffi_abi.upcast %arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
// -----

// CHECK-LABEL: tvm_ffi_abi.pack.func @cond_branch_arguments(
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.int 1
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.int 2
// CHECK: cf.cond_br %[[COND:[a-zA-Z0-9_]+]] weights([90, 10]), ^[[YES:bb[0-9]+]](%[[TRUE]] : !tvm_ffi_abi.int), ^[[NO:bb[0-9]+]](%[[FALSE]] : !tvm_ffi_abi.int)
// CHECK: ^[[YES]](%[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK: tvm_ffi_abi.upcast %[[FIRST]]
// CHECK: tvm_ffi_abi.return
// CHECK: ^[[NO]](%[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK: tvm_ffi_abi.upcast %[[SECOND]]
// CHECK: tvm_ffi_abi.return
tvm_ffi_abi.func @cond_branch_arguments() -> !tvm_ffi_abi.int {
  %cond = arith.constant true
  %true_value = tvm_ffi_abi.constant.int 1
  %false_value = tvm_ffi_abi.constant.int 2
  cf.cond_br %cond, ^true(%true_value : !tvm_ffi_abi.int),
                         ^false(%false_value : !tvm_ffi_abi.int) {branch_weights = array<i32: 90, 10>}
^true(%true_arg: !tvm_ffi_abi.int):
  %true_payload = tvm_ffi_abi.upcast %true_arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %true_ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %true_ok, %true_payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
^false(%false_arg: !tvm_ffi_abi.int):
  %false_payload = tvm_ffi_abi.upcast %false_arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %false_ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %false_ok, %false_payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
// -----

// CHECK-LABEL: tvm_ffi_abi.pack.func @explicit_status(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.pack.arguments
// CHECK: %[[RAW_FLAG:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get %[[PACK]][%[[ZERO:[a-zA-Z0-9_]+]]]
// CHECK: %[[RAW_PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get %[[PACK]][%[[ONE:[a-zA-Z0-9_]+]]]
// CHECK: tvm_ffi_abi.isa %[[RAW_FLAG]] is !tvm_ffi_abi.bool
// CHECK-NOT: tvm_ffi_abi.isa
// CHECK: cf.cond_br %[[MATCHES:[a-zA-Z0-9_]+]], ^[[ACCEPT:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[ACCEPT]]:
// CHECK: %[[TYPED_FLAG:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[RAW_FLAG]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.bool
// CHECK-NOT: tvm_ffi_abi.downcast
// CHECK: cf.br ^[[BODY:bb[0-9]+]](%[[TYPED_FLAG]], %[[RAW_PAYLOAD]] : !tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK: ^[[BODY]](%[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool, %[[PAYLOAD:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any):
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.get %[[FLAG]]
// CHECK: %[[CHECKED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi_abi.any) {
// CHECK: %[[TYPED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[PAYLOAD]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.tensor
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[TYPED]]
// CHECK: scf.yield %[[ANY]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi_abi.downcast
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.none
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[NONE]]
// CHECK: scf.yield %[[FAILED]]
// CHECK: tvm_ffi_abi.return %[[FLAG]], %[[CHECKED]]
tvm_ffi_abi.func @explicit_status(%flag: !tvm_ffi_abi.bool, %payload: !tvm_ffi_abi.any) -> !tvm_ffi_abi.tensor {
  tvm_ffi_abi.return %flag, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}

// CHECK-LABEL: tvm_ffi_abi.pack.func private @external(!tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.any
// Declarations only need their packed signature; no parameter preflight runs.
tvm_ffi_abi.func private @external(i64) -> !tvm_ffi_abi.int

// CHECK-LABEL: tvm_ffi_abi.pack.func nested @external_nested(!tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.any
tvm_ffi_abi.func nested @external_nested(!tvm_ffi_api.tensor) -> !tvm_ffi_abi.none

// CHECK-LABEL: tvm_ffi_abi.pack.func private @packed_unit(
// CHECK: scf.if
// CHECK-NOT: tvm_ffi_abi.downcast
// CHECK: tvm_ffi_abi.constant.none
// CHECK: tvm_ffi_abi.return
tvm_ffi_abi.func private @packed_unit() -> !tvm_ffi_abi.none {
  %none = tvm_ffi_abi.constant.none
  %any = tvm_ffi_abi.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
// -----

// CHECK-LABEL: tvm_ffi_abi.pack.func @forward_block_use(
// CHECK: cf.br ^[[BODY:bb[0-9]+]]
// CHECK: ^[[BODY]]:
// CHECK-NEXT: cf.br ^[[DEFINITION:bb[0-9]+]]
// CHECK: ^[[CLEANUP:bb[0-9]+]]:
// CHECK: tvm_ffi_abi.object.dec_ref %[[ARRAY:[a-zA-Z0-9_]+]] : !tvm_ffi_abi.array
// CHECK: tvm_ffi_abi.return
// CHECK: ^[[DEFINITION]]:
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.Array"
// CHECK: %[[ARRAY]] = tvm_ffi_abi.function.call %[[HANDLE]] : !tvm_ffi_abi.function(%[[ONE:[a-zA-Z0-9_]+]])
// CHECK: cf.br ^[[CLEANUP]]
tvm_ffi_abi.func @forward_block_use() -> !tvm_ffi_abi.none {
  cf.br ^definition
^cleanup:
  tvm_ffi_abi.object.dec_ref %array : !tvm_ffi_abi.array
  %unit_none = tvm_ffi_abi.constant.none
  %payload = tvm_ffi_abi.upcast %unit_none : !tvm_ffi_abi.none -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
^definition:
  %one = tvm_ffi_abi.constant.int 1
  %function = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
  %array = tvm_ffi_abi.function.call %function : !tvm_ffi_abi.function(%one) : (!tvm_ffi_abi.int) -> !tvm_ffi_abi.array
  cf.br ^cleanup
}
// -----

// CHECK-LABEL: tvm_ffi_abi.pack.func @guarded(
// CHECK-SAME: %[[PACK:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.pack.arguments
// CHECK: %[[COUNT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.size %[[PACK]]
// CHECK: %[[TWO:[a-zA-Z0-9_]+]] = arith.constant 2 : i64
// CHECK: %[[ARITY:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[COUNT]], %[[TWO]]
// CHECK-NEXT: cf.cond_br %[[ARITY]], ^[[UNPACK:bb[0-9]+]], ^[[REJECT:bb[0-9]+]]
// CHECK: ^[[UNPACK]]:
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get %[[PACK]][%[[ZERO:[a-zA-Z0-9_]+]]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.pack.get %[[PACK]][%[[ONE:[a-zA-Z0-9_]+]]]
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.isa %[[FIRST]] is !tvm_ffi_abi.int
// CHECK: %[[PREFIX:[a-zA-Z0-9_]+]] = arith.andi %[[TRUE]], %[[INT]]
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi_abi.isa %[[SECOND]] is !tvm_ffi_abi.bool
// CHECK: %[[MATCHES:[a-zA-Z0-9_]+]] = arith.andi %[[PREFIX]], %[[BOOL]]
// CHECK: cf.cond_br %[[MATCHES]], ^[[ACCEPT:bb[0-9]+]], ^[[REJECT]]
// CHECK: ^[[ACCEPT]]:
// CHECK: %[[ARG0:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[FIRST]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.int
// CHECK: %[[ARG1:[a-zA-Z0-9_]+]] = tvm_ffi_abi.downcast %[[SECOND]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.bool
// CHECK: cf.br ^[[BODY:bb[0-9]+]](%[[ARG0]], %[[ARG1]] : !tvm_ffi_abi.int, !tvm_ffi_abi.bool)
// CHECK: ^[[REJECT]]:
// CHECK: tvm_ffi_abi.exception "GuardMatch"
// CHECK: ^[[BODY]](%[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool):
// CHECK: tvm_ffi_abi.upcast %[[INPUT]]
tvm_ffi_abi.func @guarded(%arg0: !tvm_ffi_abi.int, %arg1: !tvm_ffi_abi.bool) -> !tvm_ffi_abi.int {
  %payload = tvm_ffi_abi.upcast %arg0 : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
// -----

// Functions without the old opt-in marker also become packed entries.
// CHECK-LABEL: tvm_ffi_abi.pack.func public @no_wrapper(
// CHECK: tvm_ffi_abi.constant.int 3
tvm_ffi_abi.func public @no_wrapper() -> !tvm_ffi_abi.int {
  %value = tvm_ffi_abi.constant.int 3
  %payload = tvm_ffi_abi.upcast %value : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
// -----

// CHECK-LABEL: tvm_ffi_abi.pack.func @switch_argument(
// CHECK: ^[[BODY:bb[0-9]+]](%[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[DEFAULT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[CASE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK: %[[SELECTOR:[a-zA-Z0-9_]+]], %[[DEVICE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.device %[[TENSOR]]
// CHECK: cf.switch %[[SELECTOR]] : i32, [
// CHECK: default: ^[[MERGE:bb[0-9]+]](%[[DEFAULT]] : !tvm_ffi_abi.int),
// CHECK: 1: ^[[MERGE]](%[[CASE]] : !tvm_ffi_abi.int)
// CHECK: ^[[MERGE]](%[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int):
// CHECK: tvm_ffi_abi.upcast %[[VALUE]]
tvm_ffi_abi.func @switch_argument(
    %tensor: !tvm_ffi_abi.tensor,
    %default_value: !tvm_ffi_abi.int,
    %case_value: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  %selector, %device_index = tvm_ffi_abi.tensor.device %tensor : !tvm_ffi_abi.tensor
  cf.switch %selector : i32, [
    default: ^merge(%default_value : !tvm_ffi_abi.int),
    1: ^merge(%case_value : !tvm_ffi_abi.int)
  ]
^merge(%value: !tvm_ffi_abi.int):
  %payload = tvm_ffi_abi.upcast %value : !tvm_ffi_abi.int -> !tvm_ffi_abi.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_abi.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_abi.any
}
