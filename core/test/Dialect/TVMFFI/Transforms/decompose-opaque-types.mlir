// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s
// RUN: trident-core-opt %s | trident-core-opt --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s

// CHECK-LABEL: func.func @objects(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor,
// CHECK-SAME: %[[FUNCTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function,
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.exception)
// CHECK: return %[[TENSOR]], %[[FUNCTION]], %[[EXCEPTION]] : !tvm_ffi.c.tensor, !tvm_ffi.c.function, !tvm_ffi.c.exception
func.func @objects(%tensor: !tvm_ffi.tensor, %function: !tvm_ffi.function, %exception: !tvm_ffi.exception) -> (!tvm_ffi.tensor, !tvm_ffi.function, !tvm_ffi.exception) {
  return %tensor, %function, %exception : !tvm_ffi.tensor, !tvm_ffi.function, !tvm_ffi.exception
}

// CHECK-LABEL: func.func @lookup(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function() : () -> !tvm_ffi.c.array
// CHECK: return %[[ARRAY]] : !tvm_ffi.c.array
func.func @lookup() -> !tvm_ffi.array<!tvm_ffi.any> {
  %handle = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
  %array = tvm_ffi.FunctionCall %handle : !tvm_ffi.function() : () -> !tvm_ffi.array<!tvm_ffi.any>
  return %array : !tvm_ffi.array<!tvm_ffi.any>
}

// CHECK-LABEL: func.func @strings(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// CHECK: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "hello"
// CHECK: %[[STRING:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW]] : !tvm_ffi.raw_str -> !tvm_ffi.c.any
// CHECK: return %[[STRING]], %[[VALUE]]
func.func @strings(%value: !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>) -> (!tvm_ffi.str, !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>) {
  %text = tvm_ffi.constant.str "hello"
  return %text, %value : !tvm_ffi.str, !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>
}

// CHECK-LABEL: func.func @result_and_attributes(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>)
// CHECK-SAME: attributes {test.types = [!tvm_ffi.c.exception, (!tvm_ffi.c.tensor) -> !tvm_ffi.c.function]}
// CHECK: return %[[RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
func.func @result_and_attributes(%value: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {test.types = [!tvm_ffi.exception, (!tvm_ffi.tensor) -> !tvm_ffi.function]} {
  return %value : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}
