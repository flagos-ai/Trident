// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'
// RUN: trident-core-opt %s --canonicalize --lower-tvm-ffi-defer --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'

// CHECK-LABEL: func.func @order(
// CHECK-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: tvm_ffi.ObjectIncRef %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
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
