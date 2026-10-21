// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// CHECK: llvm.func @abort()
// CHECK-LABEL: llvm.func @container_roundtrip(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[LOOKUP_STATUS:[a-zA-Z0-9_]+]], %[[LOOKUP_ZERO:[a-zA-Z0-9_]+]] : i32
// CHECK-NEXT: llvm.cond_br %[[LOOKUP_OK]], ^[[RESOLVED:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: %[[NONNULL:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[HANDLE]], %[[NULL:[a-zA-Z0-9_]+]] : !llvm.ptr
// CHECK: llvm.cond_br %[[NONNULL]]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[COUNT:[a-zA-Z0-9_]+]], %[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS]], %[[ZERO:[a-zA-Z0-9_]+]] : i32
// CHECK: llvm.cond_br %[[OK]], ^[[SUCCESS:bb[0-9]+]], ^[[CALL_FAILURE:bb[0-9]+]]
// CHECK: ^[[CALL_FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: ^[[SUCCESS]]:
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
// CHECK: llvm.store %[[RESULT]], %[[SIZE_SLOT:[a-zA-Z0-9_]+]]
// CHECK: %[[SIZE_COPY:[a-zA-Z0-9_]+]] = llvm.load %[[SIZE_SLOT]]
// CHECK-NEXT: llvm.store %[[SIZE_COPY]], %[[TUPLE_ARGS:[a-zA-Z0-9_]+]]
// CHECK: %[[CREATE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[CREATE_HANDLE:[a-zA-Z0-9_]+]], %[[TUPLE_ARGS]], %[[CREATE_COUNT:[a-zA-Z0-9_]+]], %[[TUPLE_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = llvm.load %[[TUPLE_SLOT]]
// CHECK: llvm.store %[[TUPLE]], %[[ITEM_SLOT:[a-zA-Z0-9_]+]]
// CHECK: %[[ITEM_COPY:[a-zA-Z0-9_]+]] = llvm.load %[[ITEM_SLOT]]
// CHECK-NEXT: llvm.store %[[ITEM_COPY]], %[[GET_ARGS:[a-zA-Z0-9_]+]]
// CHECK: %[[GET_STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[GET_HANDLE:[a-zA-Z0-9_]+]], %[[GET_ARGS]], %[[GET_COUNT:[a-zA-Z0-9_]+]], %[[GET_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = llvm.load %[[GET_SLOT]]
// CHECK: llvm.return %[[ITEM]]
func.func @container_roundtrip(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.int {
  %size = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  %tuple = tvm_ffi.tuple.construct(%size) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %result = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// Tuple construction packs multiple values before crossing the ABI.
// CHECK-LABEL: llvm.func @tuple_return(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Y:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[Z:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: llvm.store %[[X]], %[[CALL_ARGS:[a-zA-Z0-9_]+]]
// CHECK: %[[Y_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CALL_ARGS]][1]
// CHECK: llvm.store %[[Y]], %[[Y_SLOT]]
// CHECK: %[[Z_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CALL_ARGS]][2]
// CHECK: llvm.store %[[Z]], %[[Z_SLOT]]
// CHECK: %[[COPY_X:[a-zA-Z0-9_]+]] = llvm.load %[[CALL_ARGS]]
// CHECK: llvm.store %[[COPY_X]], %[[PACKED_ARGS:[a-zA-Z0-9_]+]]
// CHECK: %[[PACKED_Y:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[PACKED_ARGS]][1]
// CHECK: %[[COPY_Y:[a-zA-Z0-9_]+]] = llvm.load %[[Y_SLOT]]
// CHECK: llvm.store %[[COPY_Y]], %[[PACKED_Y]]
// CHECK: %[[PACKED_Z:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[PACKED_ARGS]][2]
// CHECK: %[[COPY_Z:[a-zA-Z0-9_]+]] = llvm.load %[[Z_SLOT]]
// CHECK: llvm.store %[[COPY_Z]], %[[PACKED_Z]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[PACKED_ARGS]], %[[COUNT:[a-zA-Z0-9_]+]], %[[ARRAY_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = llvm.load %[[ARRAY_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARRAY]], %[[OUT]]
// CHECK-LABEL: llvm.func @__tvm_ffi_tuple_return(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: llvm.call @tuple_return(%[[X:[a-zA-Z0-9_]+]], %[[Y:[a-zA-Z0-9_]+]], %[[Z:[a-zA-Z0-9_]+]], %[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[PACKED]], %[[OUT]]
tvm_ffi.func @tuple_return(%x: !tvm_ffi.c.tensor,
    %y: !tvm_ffi.c.tensor, %z: !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor>
    attributes {emit_tvm_ffi_abi} {
  %tuple = tvm_ffi.tuple.construct(%x, %y, %z) : (!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor>
  %return_value_2 = tvm_ffi.upcast %tuple : !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor> -> !tvm_ffi.any
  %return_flag_2 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.any
}
