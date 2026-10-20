//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s --check-prefix=DIALECT

// A semantic TVM FFI value may be returned through the generic Any ABI slot.
// DIALECT-LABEL: func.func @any_return() -> !tvm_ffi.monad.result<!tvm_ffi.any> {
// DIALECT-NEXT: [[ANY_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// DIALECT-NEXT: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast [[ANY_VALUE]] : !tvm_ffi.int -> !tvm_ffi.any
// DIALECT-NEXT: %[[RETURN:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[WIDE]] : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
// DIALECT-NEXT: return %[[RETURN]] : !tvm_ffi.monad.result<!tvm_ffi.any>
// DIALECT-NEXT: }
func.func @any_return() -> !tvm_ffi.monad.result<!tvm_ffi.any> {
  %value = "tvm_ffi.constant.int"() <{value = 1 : i64}>
      : () -> !tvm_ffi.int

  %return_cast_0 = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.any
  %returned = tvm_ffi.monad.pure %return_cast_0 : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.any>
}

// -----

// DIALECT-LABEL: func.func @function_call() -> !tvm_ffi.monad.result<!tvm_ffi.c.array> {
// DIALECT-NEXT: [[ARRAY_FUNC:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.object
// DIALECT-NEXT: [[ARRAY_RESULT:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall [[ARRAY_FUNC]]() : () -> !tvm_ffi.c.array
// DIALECT-NEXT: %[[RETURN:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure [[ARRAY_RESULT]] : (!tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// DIALECT-NEXT: return %[[RETURN]] : !tvm_ffi.monad.result<!tvm_ffi.c.array>
// DIALECT-NEXT: }
func.func @function_call() -> !tvm_ffi.monad.result<!tvm_ffi.c.array> {
  %func = tvm_ffi.FunctionGetGlobal "ffi.Array"
      : !tvm_ffi.object
  %result = tvm_ffi.FunctionCall %func()
      : () -> !tvm_ffi.c.array

  %returned = tvm_ffi.monad.pure %result : (!tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.c.array>
}

// -----

// DIALECT-LABEL: func.func @get_bool(
// DIALECT-SAME: [[ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> i1 {
// DIALECT: [[VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[ARG]] : !tvm_ffi.bool -> i1
// DIALECT-NEXT: return [[VALUE]] : i1
func.func @get_bool(%arg: !tvm_ffi.bool) -> i1 {
  %value = tvm_ffi.get %arg : !tvm_ffi.bool -> i1
  return %value : i1
}

// -----

// DIALECT-LABEL: func.func @get_float(
// DIALECT-SAME: [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.float) -> f64 {
// DIALECT-NEXT: [[FLOAT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[FLOAT_ARG]] : !tvm_ffi.float -> f64
// DIALECT-NEXT: return [[FLOAT_VALUE]] : f64
func.func @get_float(%arg: !tvm_ffi.float) -> f64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.float -> f64
  return %value : f64
}

// -----

// DIALECT-LABEL: func.func @get_int(
// DIALECT-SAME: [[INT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> i64 {
// DIALECT-NEXT: [[INT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[INT_ARG]] : !tvm_ffi.int -> i64
// DIALECT-NEXT: return [[INT_VALUE]] : i64
func.func @get_int(%arg: !tvm_ffi.int) -> i64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.int -> i64
  return %value : i64
}

// -----

// DIALECT-LABEL: func.func @get_tensor(
// DIALECT-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.object {
// DIALECT-NEXT: [[TENSOR_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[TENSOR_ARG]] : !tvm_ffi.tensor -> !tvm_ffi.object
// DIALECT-NEXT: return [[TENSOR_VALUE]] : !tvm_ffi.object
func.func @get_tensor(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %value = tvm_ffi.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  return %value : !tvm_ffi.object
}

// -----

// DIALECT-LABEL: func.func @lifetime_types(
// DIALECT-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// DIALECT-SAME: [[ARRAY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// DIALECT-SAME: [[ANY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.any,
// DIALECT-SAME: [[INDEX_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int,
// DIALECT-SAME: [[FUNCTION_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.object) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
// DIALECT: [[ITEM:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall [[FUNCTION_ARG]]([[ARRAY_ARG]], [[INDEX_ARG]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ITEM]] : !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ARRAY_ARG]] : !tvm_ffi.c.array
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: %[[RETURN:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure [[ITEM]] : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
// DIALECT-NEXT: return %[[RETURN]] : !tvm_ffi.monad.result<!tvm_ffi.tensor>
// DIALECT-NEXT: }
func.func @lifetime_types(
    %tensor: !tvm_ffi.tensor,
    %array: !tvm_ffi.c.array,
    %any: !tvm_ffi.any,
    %index: !tvm_ffi.int,
    %function: !tvm_ffi.object) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> {
  %item = tvm_ffi.FunctionCall %function(%array, %index)
      : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
  tvm_ffi.ObjectIncRef %item : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.c.array
  tvm_ffi.ObjectIncRef %any : !tvm_ffi.any
  tvm_ffi.ObjectDecRef %any : !tvm_ffi.any

  %returned = tvm_ffi.monad.pure %item : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// -----

// DIALECT-LABEL: func.func @test() -> !tvm_ffi.monad.result<!tvm_ffi.none> {
// DIALECT-NEXT: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// DIALECT-NEXT: %[[UNIT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]] : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
// DIALECT-NEXT: return %[[UNIT]] : !tvm_ffi.monad.result<!tvm_ffi.none>
// DIALECT-NEXT: }
func.func @test() -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// DIALECT-LABEL: func.func @to_native_scalars(
// DIALECT-SAME: [[BOOL_ARG:%[a-zA-Z0-9_]+]]: i1, [[INT_ARG:%[a-zA-Z0-9_]+]]: i64, [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: f64)
// DIALECT: [[BOOL:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[BOOL_ARG]] : i1 -> !tvm_ffi.bool
// DIALECT: [[INT:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[INT_ARG]] : i64 -> !tvm_ffi.int
// DIALECT: [[FLOAT:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[FLOAT_ARG]] : f64 -> !tvm_ffi.float
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
      : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}

// -----

// DIALECT-LABEL: func.func @unordered_union(
// DIALECT-SAME: [[VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
// DIALECT-NEXT: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// DIALECT-NEXT: %[[UNIT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]] : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
// DIALECT-NEXT: return %[[UNIT]] : !tvm_ffi.monad.result<!tvm_ffi.none>
// DIALECT-NEXT: }
func.func @unordered_union(
    %value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {

  %returned_none = tvm_ffi.constant.none
  %returned = tvm_ffi.monad.pure %returned_none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// -----

// DIALECT-LABEL: func.func @with_ffi_int(
// DIALECT-SAME: [[INT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
// DIALECT-NEXT: %[[RETURN:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure [[INT_ARG]] : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
// DIALECT-NEXT: return %[[RETURN]] : !tvm_ffi.monad.result<!tvm_ffi.int>
// DIALECT-NEXT: }
func.func @with_ffi_int(%arg0: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {

  %returned = tvm_ffi.monad.pure %arg0 : (!tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int>
  return %returned : !tvm_ffi.monad.result<!tvm_ffi.int>
}
