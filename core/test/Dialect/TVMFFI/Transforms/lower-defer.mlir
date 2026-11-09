// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-defer --verify-each | FileCheck %s --check-prefix=CHECK --implicit-check-not='tvm_ffi.defer'
// RUN: trident-core-opt %s --lower-tvm-ffi-defer --lower-tvm-ffi-defer --verify-each | FileCheck %s --check-prefix=BODY --implicit-check-not='tvm_ffi.defer'

// Separate defers have no relative order; operations within one body do.
// BODY-LABEL: func.func @order(
// BODY-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// BODY: tvm_ffi.ObjectDecRef %[[SECOND]]
// BODY-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-LABEL: func.func @order(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: tvm_ffi.ObjectIncRef %[[FIRST]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-DAG: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: return %[[FIRST]]
func.func @order(%first: !tvm_ffi.c.tensor, %second: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %first : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %second : !tvm_ffi.c.tensor
    tvm_ffi.ObjectIncRef %second : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  tvm_ffi.defer {
    tvm_ffi.defer_yield
  }
  tvm_ffi.ObjectIncRef %first : !tvm_ffi.c.tensor
  return %first : !tvm_ffi.c.tensor
}

// CHECK-LABEL: func.func @branches(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: scf.if %[[COND]] {
// CHECK: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: return
func.func @branches(%condition: i1, %input: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  scf.if %condition {
    tvm_ffi.defer {
      tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
      tvm_ffi.defer_yield
    }
    tvm_ffi.ObjectIncRef %input : !tvm_ffi.c.tensor
  } else {
    tvm_ffi.defer {
      tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
      tvm_ffi.defer_yield
    }
  }
  return
}

// Neither the enclosing operation nor its exit needs a dedicated adapter.
// The caller owns the input independently of the scope-local reference.
// CHECK-LABEL: func.func @alloca_scope(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = memref.alloca_scope -> (!tvm_ffi.c.tensor) {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: memref.alloca_scope.return %[[INPUT]] : !tvm_ffi.c.tensor
// CHECK: return %[[RESULT]] : !tvm_ffi.c.tensor
func.func @alloca_scope(%input: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor {
  %result = memref.alloca_scope -> !tvm_ffi.c.tensor {
    tvm_ffi.ObjectIncRef %input : !tvm_ffi.c.tensor
    tvm_ffi.defer {
      tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
      tvm_ffi.defer_yield
    }
    memref.alloca_scope.return %input : !tvm_ffi.c.tensor
  }
  return %result : !tvm_ffi.c.tensor
}

// All zero-successor terminators end the active registration, including ones
// outside the old return/yield whitelist and in downstream blocks.
// CHECK-LABEL: func.func @unreachable_exit(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: cf.br ^[[EXIT:[a-zA-Z0-9_]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: cf.br ^[[FINAL:bb[0-9]+]]
// CHECK: ^[[FINAL]]:
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]]
// CHECK-NEXT: llvm.unreachable
func.func @unreachable_exit(%input: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.br ^exit
^exit:
  llvm.unreachable
}
