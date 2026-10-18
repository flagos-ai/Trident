// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s --implicit-check-not='tvm_ffi.array.' --implicit-check-not='!tvm_ffi.' --implicit-check-not='cf.assert'

// CHECK: llvm.func @abort()
// CHECK-LABEL: llvm.func @array_size(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[LOOKUP_OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[LOOKUP_STATUS:[a-zA-Z0-9_]+]], %[[LOOKUP_ZERO:[a-zA-Z0-9_]+]] : i32
// CHECK-NEXT: llvm.cond_br %[[LOOKUP_OK]], ^[[RESOLVED:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[FAILURE]]:
// CHECK-NEXT: llvm.call @abort() : () -> ()
// CHECK-NEXT: llvm.unreachable
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[COUNT:[a-zA-Z0-9_]+]], %[[SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[OK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS]], %[[ZERO:[a-zA-Z0-9_]+]] : i32
// CHECK: llvm.cond_br %[[OK]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE]]
// CHECK: ^[[SUCCESS]]:
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[SLOT]]
func.func @array_size(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.int {
  %size = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %size : !tvm_ffi.int
}
