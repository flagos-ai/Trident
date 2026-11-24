// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s --check-prefix=CHECK --implicit-check-not='tvm_ffi_abi.defer'
// RUN: trident-core-opt %s --lower-tvm-ffi-abi-defer --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s --check-prefix=BODY --implicit-check-not='tvm_ffi_abi.defer'

// Separate defers have no relative order; operations within one body do.
// BODY-LABEL: func.func @order(
// BODY-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// BODY: tvm_ffi_abi.ObjectDecRef %[[SECOND]]
// BODY-NEXT: tvm_ffi_abi.ObjectIncRef %[[SECOND]]
// CHECK-LABEL: func.func @order(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: tvm_ffi_abi.ObjectIncRef %[[FIRST]]
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[FIRST]]
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[SECOND]]
// CHECK-DAG: tvm_ffi_abi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: return %[[FIRST]]
func.func @order(%first: !tvm_ffi_abi.tensor, %second: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor {
  tvm_ffi_abi.defer {
    tvm_ffi_abi.ObjectDecRef %first : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer_yield
  }
  tvm_ffi_abi.defer {
    tvm_ffi_abi.ObjectDecRef %second : !tvm_ffi_abi.tensor
    tvm_ffi_abi.ObjectIncRef %second : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer_yield
  }
  tvm_ffi_abi.defer {
    tvm_ffi_abi.defer_yield
  }
  tvm_ffi_abi.ObjectIncRef %first : !tvm_ffi_abi.tensor
  return %first : !tvm_ffi_abi.tensor
}

// CHECK-LABEL: func.func @branches(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: scf.if %[[COND]] {
// CHECK: tvm_ffi_abi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: return
func.func @branches(%condition: i1, %input: !tvm_ffi_abi.tensor) {
  tvm_ffi_abi.defer {
    tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer_yield
  }
  scf.if %condition {
    tvm_ffi_abi.defer {
      tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
      tvm_ffi_abi.defer_yield
    }
    tvm_ffi_abi.ObjectIncRef %input : !tvm_ffi_abi.tensor
  } else {
    tvm_ffi_abi.defer {
      tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
      tvm_ffi_abi.defer_yield
    }
  }
  return
}

// Neither the enclosing operation nor its exit needs a dedicated adapter.
// The caller owns the input independently of the scope-local reference.
// CHECK-LABEL: func.func @alloca_scope(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = memref.alloca_scope -> (!tvm_ffi_abi.tensor) {
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: memref.alloca_scope.return %[[INPUT]] : !tvm_ffi_abi.tensor
// CHECK: return %[[RESULT]] : !tvm_ffi_abi.tensor
func.func @alloca_scope(%input: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor {
  %result = memref.alloca_scope -> !tvm_ffi_abi.tensor {
    tvm_ffi_abi.ObjectIncRef %input : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer {
      tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
      tvm_ffi_abi.defer_yield
    }
    memref.alloca_scope.return %input : !tvm_ffi_abi.tensor
  }
  return %result : !tvm_ffi_abi.tensor
}

// All zero-successor terminators end the active registration, including ones
// outside the old return/yield whitelist and in downstream blocks.
// CHECK-LABEL: func.func @unreachable_exit(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: cf.br ^[[EXIT:[a-zA-Z0-9_]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: cf.br ^[[FINAL:bb[0-9]+]]
// CHECK: ^[[FINAL]]:
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: llvm.unreachable
func.func @unreachable_exit(%input: !tvm_ffi_abi.tensor) {
  tvm_ffi_abi.defer {
    tvm_ffi_abi.ObjectDecRef %input : !tvm_ffi_abi.tensor
    tvm_ffi_abi.defer_yield
  }
  cf.br ^exit
^exit:
  llvm.unreachable
}
