// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --check-prefix=DEFER

// DEFER-LABEL: func.func @aliases(
// DEFER-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER: return

// CHECK-LABEL: func.func @aliases(
// CHECK-SAME: %[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[BORROWED]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[BORROWED]]
// CHECK: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ALIAS]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ALIAS]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BORROWED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[ALIAS]], %[[ALIAS]], %[[BORROWED]]
func.func @aliases(%input: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.any, !tvm_ffi.tensor) {
  %first = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %second = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %alias = tvm_ffi.upcast %first : !tvm_ffi.tensor -> !tvm_ffi.any
  return %alias, %alias, %input : !tvm_ffi.any, !tvm_ffi.any, !tvm_ffi.tensor
}

// Base extraction and branch results keep explicit ownership transfers.
// CHECK-LABEL: func.func @base_results(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK-NEXT: %[[SELECTED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi.c.tensor)
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: scf.yield %[[ELEMENT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK-NEXT: scf.yield %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SELECTED]] : !tvm_ffi.c.tensor
func.func @base_results(%tuple: !tvm_ffi.tuple<!tvm_ffi.c.tensor>, %fallback: !tvm_ffi.c.tensor, %ready: i1) -> !tvm_ffi.c.tensor {
  %selected = scf.if %ready -> (!tvm_ffi.c.tensor) {
    %element = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.c.tensor>) -> !tvm_ffi.c.tensor
    scf.yield %element : !tvm_ffi.c.tensor
  } else {
    tvm_ffi.ObjectIncRef %fallback : !tvm_ffi.c.tensor
    scf.yield %fallback : !tvm_ffi.c.tensor
  }
  return %selected : !tvm_ffi.c.tensor
}

// Each branch exports an independent Opaque reference, including borrowed captures.
// CHECK-LABEL: func.func @branch_exports(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor,
// CHECK-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[SELECTED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi.c.tensor)
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[INPUT]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK: tvm_ffi.ObjectIncRef %[[ELEMENT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: scf.yield %[[ELEMENT]] : !tvm_ffi.c.tensor
// CHECK: tvm_ffi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK-NEXT: scf.yield %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK: tvm_ffi.ObjectIncRef %[[SELECTED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SELECTED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[SELECTED]] : !tvm_ffi.c.tensor
func.func @branch_exports(%tuple: !tvm_ffi.tuple<!tvm_ffi.tensor>, %fallback: !tvm_ffi.tensor, %ready: i1) -> !tvm_ffi.tensor {
  %selected = scf.if %ready -> (!tvm_ffi.tensor) {
    %element = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tensor
    scf.yield %element : !tvm_ffi.tensor
  } else {
    scf.yield %fallback : !tvm_ffi.tensor
  }
  return %selected : !tvm_ffi.tensor
}

// Handwritten Base values keep their existing explicit lifetime management.
// CHECK-LABEL: func.func @manual(
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.exception
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.exception
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.exception
func.func @manual() -> !tvm_ffi.c.exception {
  %value = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.c.exception
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.c.exception
  return %value : !tvm_ffi.c.exception
}

// A mixed extraction acquires independent tensor references, but no scalar reference.
// DEFER-LABEL: func.func @mixed(
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// DEFER: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// DEFER: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// DEFER: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: return
// CHECK-LABEL: func.func @mixed(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[H0:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[H0]] : !tvm_ffi.c.function(%[[TUPLE]], %[[INDEX0:[a-zA-Z0-9_]+]])
// CHECK: %[[H1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[H1]] : !tvm_ffi.c.function(%[[TUPLE]], %[[INDEX1:[a-zA-Z0-9_]+]])
// CHECK: %[[H2:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[H2]] : !tvm_ffi.c.function(%[[TUPLE]], %[[INDEX2:[a-zA-Z0-9_]+]])
// CHECK: tvm_ffi.ObjectDecRef %[[H2]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return
func.func @mixed(%tuple: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.tensor>) {
  %first, %scalar, %last = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.tensor>) -> (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.tensor)
  return
}

