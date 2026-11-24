// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-each | trident-core-opt --split-input-file --verify-each | FileCheck %s

// CHECK-LABEL: func.func @array_ops(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi_api.array.create(%[[VALUE]]) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
// CHECK: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi_api.array.get %[[ARRAY]][%[[INDEX]]] : <!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi_api.array.size %[[ARRAY]] : <!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
// CHECK: return %[[ELEMENT]], %[[SIZE]]
func.func @array_ops(%value: !tvm_ffi_abi.int, %index: i64) -> (!tvm_ffi_abi.int, !tvm_ffi_abi.int) {
  %array = tvm_ffi_api.array.create(%value) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  %element = tvm_ffi_api.array.get %array[%index] : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
  %size = tvm_ffi_api.array.size %array : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.int
  return %element, %size : !tvm_ffi_abi.int, !tvm_ffi_abi.int
}

// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct() : () -> !tvm_ffi_api.tuple<>
// CHECK: tvm_ffi_api.tuple.unconstruct %[[EMPTY]] : (!tvm_ffi_api.tuple<>) -> ()
// CHECK: return %[[EMPTY]] : !tvm_ffi_api.tuple<>
func.func @empty_tuple() -> !tvm_ffi_api.tuple<> {
  %tuple = tvm_ffi_api.tuple.construct() : () -> !tvm_ffi_api.tuple<>
  tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<>) -> ()
  return %tuple : !tvm_ffi_api.tuple<>
}

// CHECK-LABEL: func.func @pack_unpack(
// CHECK-SAME: %[[INTEGER:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int, %[[BOOLEAN:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool, %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi_api.array<!tvm_ffi_abi.int>)
// CHECK: %[[SINGLE:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[INTEGER]]) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi_api.tuple.construct(%[[SINGLE]], %[[BOOLEAN]], %[[ARRAY]])
// CHECK: %[[ITEMS:[a-zA-Z0-9_]+]]:3 = tvm_ffi_api.tuple.unconstruct %[[TUPLE]]
// CHECK: return %[[ITEMS]]#0, %[[ITEMS]]#1, %[[ITEMS]]#2
func.func @pack_unpack(%integer: !tvm_ffi_abi.int, %boolean: !tvm_ffi_abi.bool, %array: !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>) {
  %single = tvm_ffi_api.tuple.construct(%integer) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>
  %tuple = tvm_ffi_api.tuple.construct(%single, %boolean, %array) : (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>>
  %nested, %bool, %list = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>>) -> (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>)
  return %nested, %bool, %list : !tvm_ffi_api.tuple<!tvm_ffi_abi.int>, !tvm_ffi_abi.bool, !tvm_ffi_api.array<!tvm_ffi_abi.int>
}

// -----

// CHECK-LABEL: func.func @kernel_specializations(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[SIZE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int) {
// CHECK: tvm_ffi_api.kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]] : !tvm_ffi_abi.tensor #tvm_ffi_api.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %[[SIZE]] : !tvm_ffi_abi.int #tvm_ffi_api.constant_specialization<value = 4 : i64>)
func.func @kernel_specializations(%tensor: !tvm_ffi_abi.tensor,
    %size: !tvm_ffi_abi.int) {
  %one = arith.constant 1 : i64
  %launch_0 = tvm_ffi_api.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64
      threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi_abi.tensor #tvm_ffi_api.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %size : !tvm_ffi_abi.int #tvm_ffi_api.constant_specialization<value = 4 : i64>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
  func.return
}

// -----

// CHECK-LABEL: func.func @function_call() -> !tvm_ffi_abi.array {
// CHECK-NEXT: [[ARRAY_FUNC:%[a-zA-Z0-9_]+]] = tvm_ffi_api.FunctionGetGlobal "ffi.Array" : !tvm_ffi_abi.function
// CHECK-NEXT: [[ARRAY_RESULT:%[a-zA-Z0-9_]+]] = tvm_ffi_api.FunctionCall [[ARRAY_FUNC]] : !tvm_ffi_abi.function() : () -> !tvm_ffi_abi.array
// CHECK-NEXT: return [[ARRAY_RESULT]] : !tvm_ffi_abi.array
// CHECK-NEXT: }
func.func @function_call() -> !tvm_ffi_abi.array {
  %func = tvm_ffi_api.FunctionGetGlobal "ffi.Array"
     : !tvm_ffi_abi.function
  %result = tvm_ffi_api.FunctionCall %func : !tvm_ffi_abi.function()
     : () -> !tvm_ffi_abi.array

  return %result : !tvm_ffi_abi.array
}

// -----

// CHECK-LABEL: func.func @get_bool(
// CHECK-SAME: [[ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool) -> i1 {
// CHECK: [[VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi_api.get [[ARG]] : !tvm_ffi_abi.bool -> i1
// CHECK-NEXT: return [[VALUE]] : i1
func.func @get_bool(%arg: !tvm_ffi_abi.bool) -> i1 {
  %value = tvm_ffi_api.get %arg : !tvm_ffi_abi.bool -> i1
  return %value : i1
}

// -----

// CHECK-LABEL: func.func @get_float(
// CHECK-SAME: [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi_abi.float) -> f64 {
// CHECK-NEXT: [[FLOAT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi_api.get [[FLOAT_ARG]] : !tvm_ffi_abi.float -> f64
// CHECK-NEXT: return [[FLOAT_VALUE]] : f64
func.func @get_float(%arg: !tvm_ffi_abi.float) -> f64 {
  %value = tvm_ffi_api.get %arg : !tvm_ffi_abi.float -> f64
  return %value : f64
}

// -----

// CHECK-LABEL: func.func @get_int(
// CHECK-SAME: [[INT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int) -> i64 {
// CHECK-NEXT: [[INT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi_api.get [[INT_ARG]] : !tvm_ffi_abi.int -> i64
// CHECK-NEXT: return [[INT_VALUE]] : i64
func.func @get_int(%arg: !tvm_ffi_abi.int) -> i64 {
  %value = tvm_ffi_api.get %arg : !tvm_ffi_abi.int -> i64
  return %value : i64
}

// -----

// CHECK-LABEL: func.func @get_tensor(
// CHECK-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.object {
// CHECK-NEXT: [[TENSOR_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi_api.get [[TENSOR_ARG]] : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
// CHECK-NEXT: return [[TENSOR_VALUE]] : !tvm_ffi_abi.object
func.func @get_tensor(%arg: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.object {
  %value = tvm_ffi_api.get %arg : !tvm_ffi_abi.tensor -> !tvm_ffi_abi.object
  return %value : !tvm_ffi_abi.object
}

// -----

// CHECK-LABEL: func.func @is_exception(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = tvm_ffi_api.isa %[[INPUT]] is !tvm_ffi_api.exception
// CHECK: return %[[MATCH]]
func.func @is_exception(%input: !tvm_ffi_api.any) -> i1 {
  %match = tvm_ffi_api.isa %input is !tvm_ffi_api.exception : !tvm_ffi_api.any
  return %match : i1
}

// -----

// CHECK-LABEL: tvm_ffi_api.func private @private_function
tvm_ffi_api.func private @private_function() -> !tvm_ffi_abi.none {
  %none = tvm_ffi_abi.constant.none
  %value = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
