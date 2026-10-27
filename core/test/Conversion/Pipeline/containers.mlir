// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// CHECK: llvm.func @abort()
// CHECK-LABEL: llvm.func @container_roundtrip(
// CHECK-NOT: llvm.addressof @__trident_tvm_ffi_handle_ffi.Array :
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.return
func.func @container_roundtrip(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.int {
  %size = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  %tuple = tvm_ffi.tuple.construct(%size) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %result = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// Tuple construction packs multiple values before crossing the ABI.
// CHECK-LABEL: llvm.func @__tvm_ffi_tuple_return(
// CHECK-SAME: %[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[INPUTS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: %[[X:[a-zA-Z0-9_]+]] = llvm.load %[[INPUTS]]
// CHECK: %[[INPUT_Y:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[INPUTS]][1]
// CHECK: %[[Y:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Y]]
// CHECK: %[[INPUT_Z:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[INPUTS]][2]
// CHECK: %[[Z:[a-zA-Z0-9_]+]] = llvm.load %[[INPUT_Z]]
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
// CHECK: llvm.br ^[[CHECKED:bb[0-9]+]](%[[ARRAY]] : !llvm.struct<(i32, i32, i64)>)
// CHECK: ^[[CHECKED]](%[[PAYLOAD:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>):
// CHECK: llvm.cond_br %[[SUCCESS:[a-zA-Z0-9_]+]], ^[[STORE:bb[0-9]+]](%[[PAYLOAD]], %[[CODE:[a-zA-Z0-9_]+]] : !llvm.struct<(i32, i32, i64)>, i32)
// CHECK: ^[[STORE]](%[[OUTPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[STORE_CODE:[a-zA-Z0-9_]+]]: i32):
// CHECK-NEXT: llvm.store %[[OUTPUT]], %[[OUT]]
tvm_ffi.func @tuple_return(%x: !tvm_ffi.c.tensor,
    %y: !tvm_ffi.c.tensor, %z: !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor>
    attributes {emit_tvm_ffi_abi} {
  %tuple = tvm_ffi.tuple.construct(%x, %y, %z) : (!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor) -> !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor>
  %return_value_2 = tvm_ffi.upcast %tuple : !tvm_ffi.tuple<!tvm_ffi.c.tensor, !tvm_ffi.c.tensor, !tvm_ffi.c.tensor> -> !tvm_ffi.any
  %return_flag_2 = tvm_ffi.constant.bool true
  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.any
}
