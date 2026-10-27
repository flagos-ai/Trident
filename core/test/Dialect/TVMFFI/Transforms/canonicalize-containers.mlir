// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s

// CHECK-LABEL: func.func @direct_roundtrip
// CHECK-NOT: tvm_ffi.tuple.construct
// CHECK-NOT: tvm_ffi.tuple.unconstruct
// CHECK: return %{{.*}}, %{{.*}}
func.func @direct_roundtrip(%integer: !tvm_ffi.int,
    %boolean: !tvm_ffi.bool) -> (!tvm_ffi.int, !tvm_ffi.bool) {
  %tuple = tvm_ffi.tuple.construct(%integer, %boolean) :
      (!tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>
  %first, %second = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.bool>) -> (!tvm_ffi.int, !tvm_ffi.bool)
  return %first, %second : !tvm_ffi.int, !tvm_ffi.bool
}

// CHECK-LABEL: func.func @object_roundtrip
// CHECK-NOT: tvm_ffi.tuple.construct
// CHECK-NOT: tvm_ffi.tuple.unconstruct
// CHECK: return %{{.*}}
func.func @object_roundtrip(%value: !tvm_ffi.any) -> !tvm_ffi.any {
  %tuple = tvm_ffi.tuple.construct(%value) :
      (!tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.any>
  %result = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.any>) -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// The tuple remains observable through the return, so the local rewrite must
// not fire even though it also has an unconstruct user.
// CHECK-LABEL: func.func @shared_tuple
// CHECK: tvm_ffi.tuple.construct
// CHECK: tvm_ffi.tuple.unconstruct
func.func @shared_tuple(%value: !tvm_ffi.int) ->
    (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.int) {
  %tuple = tvm_ffi.tuple.construct(%value) :
      (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %result = tvm_ffi.tuple.unconstruct %tuple :
      (!tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.int
  return %tuple, %result : !tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.int
}
