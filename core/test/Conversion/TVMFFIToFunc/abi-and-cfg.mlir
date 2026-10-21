//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-func | FileCheck %s --check-prefix=CHECK

// TVMFFIToFunc preserves CFG block argument types and forwards values directly.
// The unified LLVM conversion later converts both the branch and block.
// CHECK-LABEL: func.func @branch_argument(
// CHECK-SAME: %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 7
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[VALUE]] : !tvm_ffi.int)
// CHECK: ^[[NEXT]](%[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[ARG]]
// CHECK: scf.if
// CHECK: llvm.store %[[ABI:[a-zA-Z0-9_]+]], %[[SLOT]]
// CHECK: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: func.func @__tvm_ffi_branch_argument(
// CHECK: func.call @branch_argument
tvm_ffi.func @branch_argument() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 7
  cf.br ^bb1(%value : !tvm_ffi.int)

^bb1(%arg: !tvm_ffi.int):

  %return_value_1 = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_1 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// Both edges of a conditional branch carry their own destination operands.
// CHECK-LABEL: func.func @cond_branch_arguments(
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 2
// CHECK: cf.cond_br %[[COND:[a-zA-Z0-9_]+]] weights([90, 10]), ^[[YES:bb[0-9]+]](%[[TRUE]] : !tvm_ffi.int), ^[[NO:bb[0-9]+]](%[[FALSE]] : !tvm_ffi.int)
// CHECK: ^[[YES]](%[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[FIRST]]
// CHECK: return
// CHECK: ^[[NO]](%[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// CHECK: tvm_ffi.upcast %[[SECOND]]
// CHECK: return
tvm_ffi.func @cond_branch_arguments() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %cond = arith.constant true
  %true_value = tvm_ffi.constant.int 1
  %false_value = tvm_ffi.constant.int 2
  cf.cond_br %cond, ^true(%true_value : !tvm_ffi.int),
                         ^false(%false_value : !tvm_ffi.int) {branch_weights = array<i32: 90, 10>}

^true(%true_arg: !tvm_ffi.int):

  %return_value_2 = tvm_ffi.upcast %true_arg : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_2 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.c.any
^false(%false_arg: !tvm_ffi.int):

  %return_value_3 = tvm_ffi.upcast %false_arg : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_3 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_3, %return_value_3 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// Values defined in a textually later block must be remapped before an earlier
// cleanup block is cloned.
// CHECK-LABEL: func.func @forward_block_use(
// CHECK: cf.br ^[[DEFINITION:bb[0-9]+]]
// CHECK: ^[[CLEANUP:bb[0-9]+]]:
// CHECK: tvm_ffi.ObjectDecRef %[[ARRAY:[a-zA-Z0-9_]+]] : !tvm_ffi.c.array
// CHECK: return
// CHECK: ^[[DEFINITION]]:
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[ARRAY]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[ONE:[a-zA-Z0-9_]+]])
// CHECK: cf.br ^[[CLEANUP]]
tvm_ffi.func @forward_block_use() -> !tvm_ffi.none {
  cf.br ^definition
^cleanup:
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.c.array
  %unit_none = tvm_ffi.constant.none
  %return_value_4 = tvm_ffi.upcast %unit_none : !tvm_ffi.none -> !tvm_ffi.c.any
  %return_flag_4 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_4, %return_value_4 : !tvm_ffi.bool, !tvm_ffi.c.any
^definition:
  %one = tvm_ffi.constant.int 1
  %function = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
  %array = tvm_ffi.FunctionCall %function : !tvm_ffi.c.function(%one) : (!tvm_ffi.int) -> !tvm_ffi.c.array
  cf.br ^cleanup
}

// -----

// Guard failures are represented as nested SCF regions.  The final pipeline
// lowers those regions to CFG before converting to LLVM.
// The exception constructor is reached through the cached module-level handle.
// This path emits LLVM directly and manages its own temporary references, so it
// borrows the handle: no reference counting around the call at all.
// CHECK-LABEL: func.func @guarded(
// CHECK-LABEL: func.func @__tvm_ffi_guarded(
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = llvm.load %[[ARGS:[a-zA-Z0-9_]+]]
// CHECK: %[[TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[FIRST]][0]
// CHECK: llvm.icmp "eq" %[[TAG]], %[[EXPECTED:[a-zA-Z0-9_]+]]
// CHECK: scf.if %[[MATCH:[a-zA-Z0-9_]+]]
// CHECK: func.call @guarded
// CHECK: return
tvm_ffi.func @guarded(%arg0: !tvm_ffi.int, %arg1: !tvm_ffi.bool) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {

  %return_value_5 = tvm_ffi.upcast %arg0 : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_5 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_5, %return_value_5 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// A single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
// CHECK-LABEL: func.func @make_int(
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 42
// CHECK: tvm_ffi.upcast %[[VALUE]]
// CHECK-LABEL: func.func @__tvm_ffi_make_int(
// CHECK: func.call @make_int
tvm_ffi.func @make_int() -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %0 = tvm_ffi.constant.int 42

  %return_value_6 = tvm_ffi.upcast %0 : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_6 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_6, %return_value_6 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// A wrapper is opt-in. Without the attribute, only the original function
// converted to func.func is generated.
// CHECK-LABEL: func.func @no_wrapper(
// CHECK: return
// CHECK-NOT: @__tvm_ffi_no_wrapper
tvm_ffi.func @no_wrapper() -> !tvm_ffi.int {
  %value = tvm_ffi.constant.int 3

  %return_value_7 = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_7 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_7, %return_value_7 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// Switch destinations and forwarded operands are lowered by the standard CF
// interface using the shared LLVM type converter.
// CHECK-LABEL: func.func @switch_argument(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[DEFAULT:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[CASE:[a-zA-Z0-9_]+]]: !tvm_ffi.int
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

  %return_value_8 = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.c.any

  %return_flag_8 = tvm_ffi.constant.bool true

  tvm_ffi.return %return_flag_8, %return_value_8 : !tvm_ffi.bool, !tvm_ffi.c.any
}

// -----

// CHECK-LABEL: func.func @tuple_return(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[THIRD:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[FIRST]], %[[SECOND]], %[[THIRD]])
// CHECK: tvm_ffi.upcast %[[TUPLE]]
// CHECK-LABEL: func.func @__tvm_ffi_tuple_return(
// CHECK: func.call @tuple_return
tvm_ffi.func @tuple_return(%arg0: !tvm_ffi.c.tensor, %arg1: !tvm_ffi.c.tensor, %arg2: !tvm_ffi.c.tensor) -> !tvm_ffi.c.array attributes {emit_tvm_ffi_abi} {
  %handle = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
  %tuple = tvm_ffi.FunctionCall %handle : !tvm_ffi.c.function(%arg0, %arg1, %arg2) : (!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor) -> !tvm_ffi.c.array
  tvm_ffi.ObjectDecRef %handle : !tvm_ffi.c.function
  %return_value_9 = tvm_ffi.upcast %tuple : !tvm_ffi.c.array -> !tvm_ffi.c.any
  %return_flag_9 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_9, %return_value_9 : !tvm_ffi.bool, !tvm_ffi.c.any
}
