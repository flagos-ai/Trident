// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --verify-each | FileCheck %s

// Monad type conversion supports CFG signatures independently of automatic lifetimes.
// CHECK-LABEL: func.func @forward(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CHECK: cf.br ^[[NEXT:bb[0-9]+]](%[[INPUT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CHECK: ^[[NEXT]](%[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = call @identity(%[[VALUE]]) : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK: return %[[RESULT]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
func.func @forward(%input: !tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
  cf.br ^next(%input : !tvm_ffi.monad.result<!tvm_ffi.int>)
^next(%value: !tvm_ffi.monad.result<!tvm_ffi.int>):
  %result = call @identity(%value) : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
}

func.func private @identity(!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
