// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s

// CHECK-LABEL: func.func @equal(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[RHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> i1 {
// CHECK: %[[LHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[RHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[LHS]], %[[LHS_SLOT]]
// CHECK: llvm.store %[[RHS]], %[[RHS_SLOT]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RESULT_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[RESULT_SLOT]][0, 2]
// CHECK: %[[RESULT_I64:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_PTR]] : !llvm.ptr -> i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.trunc %[[RESULT_I64]] : i64 to i32
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[EQUAL:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[RESULT]], %[[ZERO]] : i32
// CHECK: return %[[EQUAL]] : i1
func.func @equal(%lhs: !tvm_ffi.bool, %rhs: !tvm_ffi.bool) -> i1 {
  %equal = tvm_ffi.eq %lhs, %rhs : !tvm_ffi.bool, !tvm_ffi.bool
  return %equal : i1
}

// -----

// CHECK-LABEL: func.func @exception() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[MESSAGE_SIZE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(11 : i64) : i64
// CHECK: %[[MESSAGE_PTR:[a-zA-Z0-9_]+]] = llvm.alloca %[[MESSAGE_SIZE]] x i8 : (i64) -> !llvm.ptr
// CHECK: %[[MESSAGE_END_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[MESSAGE_PTR]][10] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[MESSAGE_END:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i8) : i8
// CHECK: llvm.store %[[MESSAGE_END]], %[[MESSAGE_END_PTR]] : i8, !llvm.ptr
// CHECK: %[[UNDEF:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[UNDEF]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.ptrtoint %[[MESSAGE_PTR]] : !llvm.ptr to i64
// CHECK: %[[EXCEPTION_ARG:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: llvm.store %[[EXCEPTION_ARG]], %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: %[[COPY:[a-zA-Z0-9_]+]] = llvm.load %[[SOURCE]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY]], %[[CALL_ARG:[a-zA-Z0-9_]+]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EXCEPTION_HANDLE:[a-zA-Z0-9_]+]], %[[EXCEPTION_ARGS:[a-zA-Z0-9_]+]], %[[EXCEPTION_NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EXCEPTION_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[EXCEPTION_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @exception() -> !tvm_ffi.c.exception {
  %value = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
  return %value : !tvm_ffi.c.exception
}

// -----

// CHECK-LABEL: func.func @function_refs(
// CHECK-SAME: %[[FUNCTION:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[FUNCTION]]) : (!llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[FUNCTION]]) : (!llvm.ptr) -> i32
// CHECK: return
func.func @function_refs(%function: !tvm_ffi.c.function) {
  tvm_ffi.ObjectIncRef %function : !tvm_ffi.c.function
  tvm_ffi.ObjectDecRef %function : !tvm_ffi.c.function
  return
}

// -----

// The global function is resolved once at load time into a module-level
// handle; call sites read that handle instead of repeating the lookup.
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal @__trident_tvm_ffi_handle_test.identity(#llvm.zero)
// CHECK-LABEL: func.func @global_call(
// CHECK-SAME: %[[GLOBAL_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK-NOT: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[GLOBAL_HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_test.identity : !llvm.ptr
// CHECK: %[[GLOBAL_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[RESOLVED:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[GLOBAL_HANDLE]], %[[NULL]]
// CHECK: llvm.cond_br %[[RESOLVED]], ^[[FOUND:bb[0-9]+]], ^[[MISSING:bb[0-9]+]]
// CHECK: ^[[FOUND]]:
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[GLOBAL_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: llvm.br ^[[CONTINUE:bb[0-9]+]](%[[LOOKUP_ZERO:[a-zA-Z0-9_]+]] : i32)
// CHECK: ^[[CONTINUE]](%[[LOOKUP_STATUS:[a-zA-Z0-9_]+]]: i32):
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[LOOKUP_STATUS]], %[[ZERO]]
// CHECK: llvm.cond_br %[[LOOKUP_OK]], ^[[LOOKUP_SUCCESS:bb[0-9]+]], ^[[LOOKUP_FAILURE:bb[0-9]+]]
// CHECK: ^[[LOOKUP_FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: ^[[LOOKUP_SUCCESS]]:
// CHECK: %[[HANDLE_ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[NONNULL:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[GLOBAL_HANDLE]], %[[HANDLE_ZERO]]
// CHECK: llvm.cond_br %[[NONNULL]], ^[[HANDLE_SUCCESS:bb[0-9]+]], ^[[HANDLE_FAILURE:bb[0-9]+]]
// CHECK: ^[[HANDLE_FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: ^[[HANDLE_SUCCESS]]:
// CHECK: %[[GLOBAL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[GLOBAL_ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[GLOBAL_ARGS]][0]
// CHECK: llvm.store %[[GLOBAL_ARG]], %[[GLOBAL_ARG_SLOT]]
// CHECK: %[[GLOBAL_RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[EMPTY_RESULT:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store %[[EMPTY_RESULT]], %[[GLOBAL_RESULT_SLOT]]
// CHECK: %[[GLOBAL_CALL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[GLOBAL_CALL_ARG:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[GLOBAL_CALL_ARGS]][0]
// CHECK: %[[GLOBAL_ARG_COPY:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_ARG_SLOT]]
// CHECK: llvm.store %[[GLOBAL_ARG_COPY]], %[[GLOBAL_CALL_ARG]]
// CHECK: %[[GLOBAL_NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[CALL_STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[GLOBAL_HANDLE]], %[[GLOBAL_CALL_ARGS]], %[[GLOBAL_NARGS]], %[[GLOBAL_RESULT_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[CALL_ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[CALL_OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[CALL_STATUS]], %[[CALL_ZERO]]
// CHECK: llvm.cond_br %[[CALL_OK]], ^[[CALL_SUCCESS:bb[0-9]+]], ^[[CALL_FAILURE:bb[0-9]+]]
// CHECK: ^[[CALL_FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: ^[[CALL_SUCCESS]]:
// CHECK: %[[GLOBAL_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[GLOBAL_RESULT]] : !llvm.struct<(i32, i32, i64)>
// An unresolved cached handle sets a runtime error and rejoins the status check.
// CHECK: ^[[MISSING]]:
// CHECK-NOT: TVMFFIObjectIncRef
// CHECK: llvm.call @TVMFFIErrorSetRaisedFromCStr
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-1 : i32)
// CHECK: llvm.br ^[[CONTINUE]](%[[FAILED]] : i32)
// The constructor performs the single lookup and fills the cache.
// CHECK: llvm.func internal @__trident_tvm_ffi_ctor_test.identity()
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[CTOR_NAME:[a-zA-Z0-9_]+]], %[[CTOR_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[CTOR_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[CTOR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.store %[[CTOR_HANDLE]], %[[CTOR_CACHE:[a-zA-Z0-9_]+]] : !llvm.ptr, !llvm.ptr
// The destructor drops the load-time reference.
// CHECK: llvm.func internal @__trident_tvm_ffi_dtor_test.identity()
// CHECK: %[[DTOR_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[DTOR_CACHE:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[DTOR_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: llvm.mlir.global_ctors ctors = [@__trident_tvm_ffi_ctor_test.identity], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@__trident_tvm_ffi_dtor_test.identity], priorities = [65535 : i32], data = [#llvm.zero]
func.func @global_call(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %callee = tvm_ffi.FunctionGetGlobal "test.identity"
     : !tvm_ffi.c.function
  %result = tvm_ffi.FunctionCall %callee : !tvm_ffi.c.function(%arg)
     : (!tvm_ffi.int) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @object_refs(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[INC_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][2]
// CHECK: %[[INC_HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[INC_BITS]] : i64 to !llvm.ptr
// CHECK: %[[INC_NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[INC_NONNULL:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[INC_HANDLE]], %[[INC_NULL]]
// CHECK: %[[INC_TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][0]
// CHECK: %[[INC_OBJECT_TAG:[a-zA-Z0-9_]+]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[INC_IS_OBJECT:[a-zA-Z0-9_]+]] = llvm.icmp "uge" %[[INC_TAG]], %[[INC_OBJECT_TAG]]
// CHECK: %[[INC_READY:[a-zA-Z0-9_]+]] = llvm.and %[[INC_NONNULL]], %[[INC_IS_OBJECT]]
// CHECK: llvm.cond_br %[[INC_READY]], ^[[INC_CALL:bb[0-9]+]], ^[[INC_SKIP:bb[0-9]+]]
// CHECK: ^[[INC_CALL]]:
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[INC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.br ^[[INC_SKIP]]
// CHECK: ^[[INC_SKIP]]:
// CHECK: %[[DEC_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][2]
// CHECK: %[[DEC_HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DEC_BITS]] : i64 to !llvm.ptr
// CHECK: %[[DEC_NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[DEC_NONNULL:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[DEC_HANDLE]], %[[DEC_NULL]]
// CHECK: %[[DEC_TAG:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][0]
// CHECK: %[[DEC_OBJECT_TAG:[a-zA-Z0-9_]+]] = llvm.mlir.constant(64 : i32)
// CHECK: %[[DEC_IS_OBJECT:[a-zA-Z0-9_]+]] = llvm.icmp "uge" %[[DEC_TAG]], %[[DEC_OBJECT_TAG]]
// CHECK: %[[DEC_READY:[a-zA-Z0-9_]+]] = llvm.and %[[DEC_NONNULL]], %[[DEC_IS_OBJECT]]
// CHECK: llvm.cond_br %[[DEC_READY]], ^[[DEC_CALL:bb[0-9]+]], ^[[DEC_SKIP:bb[0-9]+]]
// CHECK: ^[[DEC_CALL]]:
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[DEC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.br ^[[DEC_SKIP]]
// CHECK: ^[[DEC_SKIP]]:
// CHECK: return
func.func @object_refs(%object: !tvm_ffi.c.tensor) {
  tvm_ffi.ObjectIncRef %object : !tvm_ffi.c.tensor
  tvm_ffi.ObjectDecRef %object : !tvm_ffi.c.tensor
  return
}

// -----

// CHECK-LABEL: func.func @clone(
// CHECK-SAME: %[[CLONE_INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[CLONE_SIZE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_INPUTS:[a-zA-Z0-9_]+]] = llvm.alloca %[[CLONE_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[CLONE_INPUT]], %[[CLONE_INPUTS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_CALL_INPUTS:[a-zA-Z0-9_]+]] = llvm.alloca %[[CLONE_CALL_SIZE:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_CALL_INPUT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CLONE_CALL_INPUTS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[CLONE_INPUT_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_INPUTS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_INPUT_VALUE]], %[[CLONE_CALL_INPUT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[CLONE_HANDLE]], %[[CLONE_CALL_INPUTS]], %[[CLONE_NARGS:[a-zA-Z0-9_]+]], %[[CLONE_RESULTS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[CLONE_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_RESULTS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[CLONE_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @clone(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor {
  %copy = tvm_ffi.tensor.clone %input : !tvm_ffi.c.tensor -> !tvm_ffi.c.tensor
  return %copy : !tvm_ffi.c.tensor
}

// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[COPY_DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[COPY_SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[COPY_SIZE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_DEST_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_SOURCE_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[COPY_DESTINATION]], %[[COPY_DEST_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.store %[[COPY_SOURCE]], %[[COPY_SOURCE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_ARGUMENTS_COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[COPY_ARGUMENTS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_ARGUMENTS_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_DESTINATION_ARGUMENT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[COPY_ARGUMENTS_SLOT]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[COPY_DESTINATION_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_DEST_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_DESTINATION_VALUE]], %[[COPY_DESTINATION_ARGUMENT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_SOURCE_ARGUMENT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[COPY_ARGUMENTS_SLOT]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[COPY_SOURCE_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_SOURCE_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_SOURCE_VALUE]], %[[COPY_SOURCE_ARGUMENT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[COPY_HANDLE]], %[[COPY_ARGUMENTS_SLOT]], %[[COPY_NUM_ARGS:[a-zA-Z0-9_]+]], %[[COPY_RESULTS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: return
func.func @copy(%destination: !tvm_ffi.c.tensor, %source: !tvm_ffi.c.tensor) {
  tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.c.tensor
  return
}
