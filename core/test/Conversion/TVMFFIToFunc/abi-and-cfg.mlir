//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -lower-tvm-ffi-monad -convert-tvm-ffi-to-func | FileCheck %s --check-prefixes=INTERMEDIATE,FUNC

// Guard failures are represented as nested SCF regions.  The final pipeline
// lowers those regions to CFG before converting to LLVM.
// INTERMEDIATE-LABEL: func.func @__tvm_ffi_guarded(
// INTERMEDIATE-SAME: [[CONTEXT:%[a-zA-Z0-9_]+]]: !llvm.ptr, [[ARGS:%[a-zA-Z0-9_]+]]: !llvm.ptr, [[NARGS:%[a-zA-Z0-9_]+]]: i32, [[RESULT:%[a-zA-Z0-9_]+]]: !llvm.ptr)
// INTERMEDIATE: [[GUARD_TRUE:%[a-zA-Z0-9_]+]] = arith.constant true
// INTERMEDIATE-NEXT: [[GUARD_INT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_INT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[GUARD_INT_ANY]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.int
// INTERMEDIATE-NEXT: [[GUARD_INT_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[GUARD_INT_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// INTERMEDIATE-NEXT: [[GUARD_INT_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[GUARD_INT_TYPE]], [[GUARD_INT_KIND]] : i32
// INTERMEDIATE-NEXT: [[GUARD_AFTER_INT:%[a-zA-Z0-9_]+]] = arith.andi [[GUARD_TRUE]], [[GUARD_INT_VALID]] : i1
// INTERMEDIATE-NEXT: [[GUARD_BOOL_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_BOOL_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[GUARD_BOOL_ANY]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.bool
// INTERMEDIATE-NEXT: [[GUARD_BOOL_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[GUARD_BOOL_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// INTERMEDIATE-NEXT: [[GUARD_BOOL_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[GUARD_BOOL_TYPE]], [[GUARD_BOOL_KIND]] : i32
// INTERMEDIATE-NEXT: [[ALL_GUARDS_VALID:%[a-zA-Z0-9_]+]] = arith.andi [[GUARD_AFTER_INT]], [[GUARD_BOOL_VALID]] : i1
// INTERMEDIATE-NEXT: %[[WRAPPER_STATUS:[a-zA-Z0-9_]+]] = scf.if [[ALL_GUARDS_VALID]] -> (i32) {
// INTERMEDIATE: [[GUARDED_STATUS:%[a-zA-Z0-9_]+]] = func.call @guarded([[GUARD_INT]], [[GUARD_BOOL]], [[OUTPUT_SLOT:%[a-zA-Z0-9_]+]]) : (!tvm_ffi.int, !tvm_ffi.bool, !llvm.ptr) -> i32
// INTERMEDIATE: [[LOADED:%[a-zA-Z0-9_]+]] = llvm.load [[OUTPUT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: llvm.store [[LOADED]], [[RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// The exception constructor is reached through the cached module-level handle.
// This path emits LLVM directly and is never seen by the ownership pass, so it
// borrows the handle: no reference counting around the call at all.
// INTERMEDIATE: [[GUARD_EXCEPTION_HANDLE_ADDR:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.ffi.Exception : !llvm.ptr
// INTERMEDIATE-NEXT: [[GUARD_EXCEPTION_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_EXCEPTION_HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// INTERMEDIATE-NOT: llvm.call @TVMFFIObjectIncRef
// INTERMEDIATE: [[GUARD_EXCEPTION_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[GUARD_EXCEPTION_HANDLE]], [[GUARD_EXCEPTION_ARGS:%[a-zA-Z0-9_]+]], [[GUARD_EXCEPTION_NARGS:%[a-zA-Z0-9_]+]], [[GUARD_FAILURE_RESULT_SLOT:%[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// INTERMEDIATE-NOT: llvm.call @TVMFFIObjectDecRef
// INTERMEDIATE-NEXT: [[GUARD_FAILURE_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_FAILURE_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: llvm.store [[GUARD_FAILURE_RESULT]], [[RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE-NEXT: %[[GUARD_STATUS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// INTERMEDIATE-NEXT: scf.yield %[[GUARD_STATUS]] : i32
// INTERMEDIATE-NEXT: }
// INTERMEDIATE-NEXT: return %[[WRAPPER_STATUS]] : i32
tvm_ffi.func @guarded(%arg0: !tvm_ffi.int, %arg1: !tvm_ffi.bool) -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {

  %returned = tvm_ffi.monad.pure %arg0 : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// TVMFFIToFunc preserves CFG block argument types and forwards values directly.
// The unified LLVM conversion later converts both the branch and block.
// INTERMEDIATE-LABEL: func.func @branch_argument(
// INTERMEDIATE: [[BRANCH_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 7
// INTERMEDIATE-NEXT: cf.br [[BRANCH_DEST:\^bb[0-9]+]]([[BRANCH_VALUE]] : !tvm_ffi.int)
// INTERMEDIATE: [[BRANCH_DEST]]([[BRANCH_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// INTERMEDIATE: %[[BOXED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[BRANCH_BLOCK_ARG]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOXED]], %[[SLOT:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @branch_argument() -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 7
  cf.br ^bb1(%value : !tvm_ffi.int)

^bb1(%arg: !tvm_ffi.int):

  %returned = tvm_ffi.monad.pure %arg : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// Both edges of a conditional branch carry their own destination operands.
// INTERMEDIATE-LABEL: func.func @cond_branch_arguments(
// INTERMEDIATE: [[COND:%[a-zA-Z0-9_]+]] = arith.constant true
// INTERMEDIATE-NEXT: [[TRUE_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// INTERMEDIATE-NEXT: [[FALSE_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 2
// INTERMEDIATE-NEXT: cf.cond_br [[COND]] weights([90, 10]), [[TRUE_DEST:\^bb[0-9]+]]([[TRUE_VALUE]] : !tvm_ffi.int), [[FALSE_DEST:\^bb[0-9]+]]([[FALSE_VALUE]] : !tvm_ffi.int)
// INTERMEDIATE: [[TRUE_DEST]]([[TRUE_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// INTERMEDIATE: %[[BOXED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TRUE_BLOCK_ARG]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOXED]], %[[SLOT:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// INTERMEDIATE: [[FALSE_DEST]]([[FALSE_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// INTERMEDIATE: %[[BOXED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[FALSE_BLOCK_ARG]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOXED]], %[[SLOT:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @cond_branch_arguments() -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {
  %cond = arith.constant true
  %true_value = tvm_ffi.constant.int 1
  %false_value = tvm_ffi.constant.int 2
  cf.cond_br %cond, ^true(%true_value : !tvm_ffi.int),
                         ^false(%false_value : !tvm_ffi.int) {branch_weights = array<i32: 90, 10>}

^true(%true_arg: !tvm_ffi.int):

  %returned = tvm_ffi.monad.pure %true_arg : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
^false(%false_arg: !tvm_ffi.int):

  %return_pure_1 = tvm_ffi.monad.pure %false_arg : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %return_pure_1 : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// Values defined in a textually later block must be remapped before an earlier
// cleanup block is cloned.
// INTERMEDIATE-LABEL: func.func @forward_block_use(
// INTERMEDIATE: tvm_ffi.ObjectDecRef [[ARRAY:%[a-zA-Z0-9_]+]] : !tvm_ffi.array
// INTERMEDIATE: [[ARRAY:%[a-zA-Z0-9_]+]], [[CALL_SUCCESS:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
tvm_ffi.func @forward_block_use() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  cf.br ^definition
^cleanup:
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.array
  %unit_none = tvm_ffi.constant.none
  %unit = tvm_ffi.monad.pure %unit_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %unit : !tvm_ffi.monad.result<!tvm_ffi.none>
^definition:
  %one = tvm_ffi.constant.int 1
  %function, %get_success = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function, i1
  %array, %call_success = tvm_ffi.FunctionCall %function(%one) : (!tvm_ffi.int) -> !tvm_ffi.array, i1
  cf.br ^cleanup
}

// -----

// INTERMEDIATE-LABEL: func.func @make_int(
// INTERMEDIATE: return %[[SCALAR_STATUS:[a-zA-Z0-9_]+]] : i32
// INTERMEDIATE-LABEL: func.func @__tvm_ffi_make_int(
// INTERMEDIATE: [[SCALAR_STATUS:%[a-zA-Z0-9_]+]] = func.call @make_int([[SCALAR_SLOT:%[a-zA-Z0-9_]+]]) : (!llvm.ptr) -> i32
// INTERMEDIATE: llvm.load [[SCALAR_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// A single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
tvm_ffi.func @make_int() -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {
  %0 = tvm_ffi.constant.int 42

  %returned = tvm_ffi.monad.pure %0 : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// INTERMEDIATE-LABEL: func.func @multi_return_three_tensors(
// INTERMEDIATE-SAME: %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[Y:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[Z:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// INTERMEDIATE: %[[BOX_X:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[X]] : !tvm_ffi.tensor to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOX_X]], %[[SLOT_X:[a-zA-Z0-9_]+]]
// INTERMEDIATE: %[[BOX_Y:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[Y]] : !tvm_ffi.tensor to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOX_Y]], %[[SLOT_Y:[a-zA-Z0-9_]+]]
// INTERMEDIATE: %[[BOX_Z:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[Z]] : !tvm_ffi.tensor to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOX_Z]], %[[SLOT_Z:[a-zA-Z0-9_]+]]
// INTERMEDIATE-LABEL: func.func @__tvm_ffi_multi_return_three_tensors(
// INTERMEDIATE: func.call @multi_return_three_tensors
// Multi-results are packed into an FFI Array and stored into the result slot.
// The input tensors are borrowed from the caller, so only the temporary
// ffi.Array function handle is released here; no element DecRef is emitted.
tvm_ffi.func @multi_return_three_tensors(%arg0: !tvm_ffi.tensor, %arg1: !tvm_ffi.tensor, %arg2: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>) attributes {emit_tvm_ffi_abi} {

  %returned:3 = tvm_ffi.monad.pure %arg0, %arg1, %arg2 : (!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
  tvm_ffi.return %returned#0, %returned#1, %returned#2 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// -----

// A wrapper is opt-in. Without the attribute, only the original function
// converted to func.func is generated.
// FUNC-LABEL: func.func @no_wrapper
// FUNC-NOT: func.func @__tvm_ffi_no_wrapper
// FUNC: [[NO_WRAPPER_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 3
// FUNC: %[[BOX:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[NO_WRAPPER_VALUE]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// FUNC: llvm.store %[[BOX]], %[[SLOT:[a-zA-Z0-9_]+]]
// FUNC: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @no_wrapper() -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %value = tvm_ffi.constant.int 3

  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

// -----

// Switch destinations and forwarded operands are lowered by the standard CF
// interface using the shared LLVM type converter.
// INTERMEDIATE-LABEL: func.func @switch_argument(
// INTERMEDIATE-SAME: [[SWITCH_TENSOR:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, [[DEFAULT_VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.int, [[CASE_VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// INTERMEDIATE: [[SELECTOR:%[a-zA-Z0-9_]+]], [[DEVICE_INDEX:%[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device [[SWITCH_TENSOR]] : !tvm_ffi.tensor
// INTERMEDIATE-NEXT: cf.switch [[SELECTOR]] : i32, [
// INTERMEDIATE-NEXT: default: [[SWITCH_MERGE:\^bb[0-9]+]]([[DEFAULT_VALUE]] : !tvm_ffi.int),
// INTERMEDIATE-NEXT: 1: [[SWITCH_MERGE]]([[CASE_VALUE]] : !tvm_ffi.int)
// INTERMEDIATE-NEXT: ]
// INTERMEDIATE: [[SWITCH_MERGE]]([[SWITCH_RESULT:%[a-zA-Z0-9_]+]]: !tvm_ffi.int):
// INTERMEDIATE: %[[BOXED:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[SWITCH_RESULT]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE: llvm.store %[[BOXED]], %[[SLOT:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE: return %[[STATUS:[a-zA-Z0-9_]+]] : i32
tvm_ffi.func @switch_argument(
    %tensor: !tvm_ffi.tensor,
    %default_value: !tvm_ffi.int,
    %case_value: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  %selector, %device_index = tvm_ffi.tensor.device %tensor : !tvm_ffi.tensor
  cf.switch %selector : i32, [
    default: ^merge(%default_value : !tvm_ffi.int),
    1: ^merge(%case_value : !tvm_ffi.int)
  ]

^merge(%value: !tvm_ffi.int):

  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}
