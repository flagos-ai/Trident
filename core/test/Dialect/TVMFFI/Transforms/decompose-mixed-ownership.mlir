// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --check-prefix=DEFER
// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --check-prefix=CLEANUP --implicit-check-not='tvm_ffi.defer'

// Opaque and Base tensor results have the same converted representation, but
// only the original Opaque results receive automatic local releases. Exports
// retain each Opaque occurrence before any local release, including duplicates.
// DEFER-LABEL: func.func @mixed_results(
// DEFER-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// DEFER: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FIRST_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[FIRST_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[MANUAL_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[MANUAL_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[MANUAL_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[SCALAR_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SCALAR_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[SCALAR_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[LAST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[LAST_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[LAST_INDEX:[a-zA-Z0-9_]+]])
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[LAST_HANDLE]] : !tvm_ffi.c.function
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[MANUAL]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[LAST]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
// CLEANUP-LABEL: func.func @mixed_results(
// CLEANUP: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CLEANUP: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CLEANUP: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CLEANUP: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CLEANUP: tvm_ffi.ObjectDecRef %[[MANUAL]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[LAST]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
func.func @mixed_results(%tuple: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any>) -> (!tvm_ffi.tensor, !tvm_ffi.any, !tvm_ffi.tensor, !tvm_ffi.int) {
  %first, %manual, %scalar, %last = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any>) -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any)
  tvm_ffi.ObjectDecRef %manual : !tvm_ffi.c.tensor
  return %first, %last, %first, %scalar : !tvm_ffi.tensor, !tvm_ffi.any, !tvm_ffi.tensor, !tvm_ffi.int
}
