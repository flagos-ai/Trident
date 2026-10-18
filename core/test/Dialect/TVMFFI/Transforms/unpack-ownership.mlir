// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s -lower-tvm-ffi-monad -convert-scf-to-cf -ownership-deallocation -verify-each | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @array(
// CHECK: cf.cond_br %[[READY:[a-zA-Z0-9_]+]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[SUCCESS]]:
// CHECK: tvm_ffi.ObjectIncRef %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: cf.br
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[PAYLOAD:[a-zA-Z0-9_]+]] : !tvm_ffi.array -> !tvm_ffi.any
// CHECK: cf.br
// CHECK: ^[[FAILURE]]:
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: cf.br
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.any
// CHECK: tvm_ffi.ObjectDecRef %[[SOURCE]]
// CHECK: tvm_ffi.return
tvm_ffi.func @array(%input: !tvm_ffi.monad.result<!tvm_ffi.array>) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.array>) -> (i1, !tvm_ffi.any)
  %result = tvm_ffi.monad.pure %value : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.any>
}

// CHECK-LABEL: tvm_ffi.func @exception(
// CHECK: cf.cond_br %[[READY:[a-zA-Z0-9_]+]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[SUCCESS]]:
// CHECK: tvm_ffi.ObjectIncRef %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: cf.br
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[PAYLOAD:[a-zA-Z0-9_]+]] : !tvm_ffi.exception -> !tvm_ffi.any
// CHECK: cf.br
// CHECK: ^[[FAILURE]]:
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: cf.br
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.any
// CHECK: tvm_ffi.ObjectDecRef %[[SOURCE]]
// CHECK: tvm_ffi.return
tvm_ffi.func @exception(%input: !tvm_ffi.monad.result<!tvm_ffi.exception>) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.exception>) -> (i1, !tvm_ffi.any)
  %result = tvm_ffi.monad.pure %value : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.any>
}

// CHECK-LABEL: tvm_ffi.func @tensor(
// CHECK: cf.cond_br %[[READY:[a-zA-Z0-9_]+]], ^[[SUCCESS:bb[0-9]+]], ^[[FAILURE:bb[0-9]+]]
// CHECK: ^[[SUCCESS]]:
// CHECK: tvm_ffi.ObjectIncRef %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: cf.br
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[PAYLOAD:[a-zA-Z0-9_]+]] : !tvm_ffi.tensor -> !tvm_ffi.any
// CHECK: cf.br
// CHECK: ^[[FAILURE]]:
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NONE]]
// CHECK: cf.br
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.any
// CHECK: tvm_ffi.ObjectDecRef %[[SOURCE]]
// CHECK: tvm_ffi.return
tvm_ffi.func @tensor(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  %result = tvm_ffi.monad.pure %value : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.any>
}

