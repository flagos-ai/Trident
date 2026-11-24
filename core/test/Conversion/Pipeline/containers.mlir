// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi_'  --implicit-check-not='unrealized_conversion_cast'

// CHECK: llvm.func @abort()
// CHECK-LABEL: llvm.func @container_roundtrip(
// CHECK-NOT: llvm.addressof @__trident_tvm_ffi_handle_ffi.Array :
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.return
func.func @container_roundtrip(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int {
  %size = tvm_ffi_api.array.size %array : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
  %tuple = tvm_ffi_api.tuple.construct(%size) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
  %result = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
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
// CHECK: llvm.store %[[ARRAY]], %[[OUT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi_api.func @tuple_return(%x: !tvm_ffi_abi.tensor,
    %y: !tvm_ffi_abi.tensor, %z: !tvm_ffi_abi.tensor) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor>
 {
  %tuple = tvm_ffi_api.tuple.construct(%x, %y, %z) : (!tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor>
  %return_value_2 = tvm_ffi_api.upcast %tuple : !tvm_ffi_api.tuple<!tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.tensor> -> !tvm_ffi_api.any
  %return_flag_2 = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %return_flag_2, %return_value_2 : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
