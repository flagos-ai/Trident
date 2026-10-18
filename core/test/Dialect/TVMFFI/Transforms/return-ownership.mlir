// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-scf-to-cf --ownership-deallocation --verify-each | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @borrowed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[INPUT]], %[[TRUE]]
// CHECK: scf.if %[[TRUE]] {
// CHECK: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
// CHECK: }
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK: tvm_ffi.return %[[PAYLOAD]]
tvm_ffi.func @borrowed(%input: !tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %payload = tvm_ffi.monad.pure %input : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>

  tvm_ffi.return %payload : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// Failed Any payloads carry no reference credit.
// CHECK-LABEL: tvm_ffi.func @failed_any()
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[ANY:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ZERO]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.any
// CHECK-NOT: tvm_ffi.ObjectDecRef
// CHECK: tvm_ffi.return %[[RESULT:[a-zA-Z0-9_]+]]
tvm_ffi.func @failed_any() -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.any>

  tvm_ffi.return %failed : !tvm_ffi.monad.result<!tvm_ffi.any>
}

// CHECK-LABEL: tvm_ffi.func @mixed_failure(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>)
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[FALSE:[a-zA-Z0-9_]+]] = arith.constant false
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ZERO]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.none
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[CLONE]], %[[TRUE]] : !tvm_ffi.tensor, i1 to !tvm_ffi.monad.result<!tvm_ffi.tensor>
// CHECK: %[[UNIT:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[NONE]], %[[FALSE]] : !tvm_ffi.none, i1 to !tvm_ffi.monad.result<!tvm_ffi.none>
// CHECK: %[[READY:[a-zA-Z0-9_]+]] = arith.andi %[[TRUE]], %[[FALSE]] : i1
// CHECK: scf.if %[[READY]] {
// CHECK: tvm_ffi.ObjectIncRef %[[CLONE]] : !tvm_ffi.tensor
// CHECK: }
// CHECK: tvm_ffi.ObjectDecRef %[[CLONE]] : !tvm_ffi.tensor
// CHECK: tvm_ffi.return %[[PAYLOAD]], %[[UNIT]]
tvm_ffi.func @mixed_failure(%input: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>) {
  %clone = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %payload = tvm_ffi.monad.pure %clone : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  %unit = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>

  tvm_ffi.return %payload, %unit : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.none>
}

// Repeated results represent independent exported references to one object.
// CHECK-LABEL: tvm_ffi.func @repeated(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[INPUT]], %[[TRUE:[a-zA-Z0-9_]+]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[INPUT]], %[[TRUE]]
// CHECK: scf.if %[[READY:[a-zA-Z0-9_]+]] {
// CHECK: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
// CHECK: tvm_ffi.return %[[FIRST]], %[[SECOND]]
tvm_ffi.func @repeated(%input: !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>) {
  %result = tvm_ffi.monad.pure %input : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>

  tvm_ffi.return %result, %result : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
