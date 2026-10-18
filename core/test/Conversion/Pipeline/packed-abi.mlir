// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// Scalar results are boxed by the graph and forwarded into the packed result.
// CHECK-LABEL: llvm.func @make_int(
// CHECK-SAME: %[[SCALAR_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(42 : i64)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], %[[PREFIX:[a-zA-Z0-9_]+]][2]
// CHECK: %[[SELECTED:[a-zA-Z0-9_]+]] = llvm.select %[[OK:[a-zA-Z0-9_]+]], %[[RESULT]], %[[NONE:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[SELECTED]], %[[SCALAR_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.return %[[STATUS:[a-zA-Z0-9_]+]] : i32
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[RESULT_SLOT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @make_int(%[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
// CHECK: llvm.store %[[BOXED]], %[[RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_int() -> !tvm_ffi.monad.result<!tvm_ffi.int> attributes {emit_tvm_ffi_abi} {
  %value = tvm_ffi.constant.int 42

  %returned = tvm_ffi.monad.pure %value : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}

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
// CHECK: %[[SELECTED:[a-zA-Z0-9_]+]] = llvm.select %[[OK:[a-zA-Z0-9_]+]], %[[ARRAY]], %[[NONE:[a-zA-Z0-9_]+]]
// CHECK: llvm.store %[[SELECTED]], %[[OUT]]
// CHECK-LABEL: llvm.func @__tvm_ffi_tuple_return(
// CHECK-SAME: %[[CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[NARGS:[a-zA-Z0-9_]+]]: i32, %[[OUT:[a-zA-Z0-9_]+]]: !llvm.ptr
// CHECK: llvm.call @tuple_return(%[[X:[a-zA-Z0-9_]+]], %[[Y:[a-zA-Z0-9_]+]], %[[Z:[a-zA-Z0-9_]+]], %[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[PACKED]], %[[OUT]]
tvm_ffi.func @tuple_return(%x: !tvm_ffi.tensor,
    %y: !tvm_ffi.tensor, %z: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor>>
    attributes {emit_tvm_ffi_abi} {
  %tuple = tvm_ffi.tuple.construct(%x, %y, %z) : (!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor>
  %returned = tvm_ffi.monad.pure %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor>>
  tvm_ffi.return %returned : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.tensor>>
}
