// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-each | trident-core-opt --split-input-file --verify-each | FileCheck %s

// CHECK-LABEL: func.func @array_ops(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.array.create(%[[VALUE]]) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
// CHECK: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.array.get %[[ARRAY]][%[[INDEX]]] : <!tvm_ffi.int> -> !tvm_ffi.int
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.array.size %[[ARRAY]] : <!tvm_ffi.int> -> !tvm_ffi.int
// CHECK: return %[[ELEMENT]], %[[SIZE]]
func.func @array_ops(%value: !tvm_ffi.int, %index: i64) -> (!tvm_ffi.int, !tvm_ffi.int) {
  %array = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
  %element = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  %size = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
  return %element, %size : !tvm_ffi.int, !tvm_ffi.int
}

// CHECK-LABEL: func.func @empty_tuple(
// CHECK: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
// CHECK: tvm_ffi.tuple.unconstruct %[[EMPTY]] : (!tvm_ffi.tuple<>) -> ()
// CHECK: return %[[EMPTY]] : !tvm_ffi.tuple<>
func.func @empty_tuple() -> !tvm_ffi.tuple<> {
  %tuple = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<>) -> ()
  return %tuple : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @pack_unpack(
// CHECK-SAME: %[[INTEGER:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[BOOLEAN:[a-zA-Z0-9_]+]]: !tvm_ffi.bool, %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array<!tvm_ffi.int>)
// CHECK: %[[SINGLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[INTEGER]]) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.tuple.construct(%[[SINGLE]], %[[BOOLEAN]], %[[ARRAY]])
// CHECK: %[[ITEMS:[a-zA-Z0-9_]+]]:3 = tvm_ffi.tuple.unconstruct %[[TUPLE]]
// CHECK: return %[[ITEMS]]#0, %[[ITEMS]]#1, %[[ITEMS]]#2
func.func @pack_unpack(%integer: !tvm_ffi.int, %boolean: !tvm_ffi.bool, %array: !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>) {
  %single = tvm_ffi.tuple.construct(%integer) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int>
  %tuple = tvm_ffi.tuple.construct(%single, %boolean, %array) : (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>>
  %nested, %bool, %list = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>>) -> (!tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>)
  return %nested, %bool, %list : !tvm_ffi.tuple<!tvm_ffi.int>, !tvm_ffi.bool, !tvm_ffi.array<!tvm_ffi.int>
}

// -----

// CHECK-LABEL: func.func @captured(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: tvm_ffi.defer {
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @captured(%input: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %input : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  return
}

// Even an empty defer is conservatively effectful until explicitly lowered.
// CHECK-LABEL: func.func @empty()
// CHECK: tvm_ffi.defer {
// CHECK-NEXT: tvm_ffi.defer_yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @empty() {
  tvm_ffi.defer {
    tvm_ffi.defer_yield
  }
  return
}

// -----

// -----

// CHECK-LABEL: func.func @kernel_specializations(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SIZE:[a-zA-Z0-9_]+]]: !tvm_ffi.int) {
// CHECK: tvm_ffi.kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]] : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %[[SIZE]] : !tvm_ffi.int #tvm_ffi.constant_specialization<value = 4 : i64>)
func.func @kernel_specializations(%tensor: !tvm_ffi.c.tensor,
    %size: !tvm_ffi.int) {
  %one = arith.constant 1 : i64
  %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64
      threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %size : !tvm_ffi.int #tvm_ffi.constant_specialization<value = 4 : i64>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  func.return
}

// -----

// CHECK-LABEL: func.func @function_call() -> !tvm_ffi.c.array {
// CHECK-NEXT: [[ARRAY_FUNC:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK-NEXT: [[ARRAY_RESULT:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall [[ARRAY_FUNC]] : !tvm_ffi.c.function() : () -> !tvm_ffi.c.array
// CHECK-NEXT: return [[ARRAY_RESULT]] : !tvm_ffi.c.array
// CHECK-NEXT: }
func.func @function_call() -> !tvm_ffi.c.array {
  %func = tvm_ffi.FunctionGetGlobal "ffi.Array"
     : !tvm_ffi.c.function
  %result = tvm_ffi.FunctionCall %func : !tvm_ffi.c.function()
     : () -> !tvm_ffi.c.array

  return %result : !tvm_ffi.c.array
}

// -----

// CHECK-LABEL: func.func @get_bool(
// CHECK-SAME: [[ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> i1 {
// CHECK: [[VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[ARG]] : !tvm_ffi.bool -> i1
// CHECK-NEXT: return [[VALUE]] : i1
func.func @get_bool(%arg: !tvm_ffi.bool) -> i1 {
  %value = tvm_ffi.get %arg : !tvm_ffi.bool -> i1
  return %value : i1
}

// -----

// CHECK-LABEL: func.func @get_float(
// CHECK-SAME: [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.float) -> f64 {
// CHECK-NEXT: [[FLOAT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[FLOAT_ARG]] : !tvm_ffi.float -> f64
// CHECK-NEXT: return [[FLOAT_VALUE]] : f64
func.func @get_float(%arg: !tvm_ffi.float) -> f64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.float -> f64
  return %value : f64
}

// -----

// CHECK-LABEL: func.func @get_int(
// CHECK-SAME: [[INT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> i64 {
// CHECK-NEXT: [[INT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[INT_ARG]] : !tvm_ffi.int -> i64
// CHECK-NEXT: return [[INT_VALUE]] : i64
func.func @get_int(%arg: !tvm_ffi.int) -> i64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.int -> i64
  return %value : i64
}

// -----

// CHECK-LABEL: func.func @get_tensor(
// CHECK-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.object {
// CHECK-NEXT: [[TENSOR_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[TENSOR_ARG]] : !tvm_ffi.c.tensor -> !tvm_ffi.object
// CHECK-NEXT: return [[TENSOR_VALUE]] : !tvm_ffi.object
func.func @get_tensor(%arg: !tvm_ffi.c.tensor) -> !tvm_ffi.object {
  %value = tvm_ffi.get %arg : !tvm_ffi.c.tensor -> !tvm_ffi.object
  return %value : !tvm_ffi.object
}

// -----

// CHECK-LABEL: func.func @to_native_scalars(
// CHECK-SAME: [[BOOL_ARG:%[a-zA-Z0-9_]+]]: i1, [[INT_ARG:%[a-zA-Z0-9_]+]]: i64, [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: f64)
// CHECK: [[BOOL:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[BOOL_ARG]] : i1 -> !tvm_ffi.bool
// CHECK: [[INT:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[INT_ARG]] : i64 -> !tvm_ffi.int
// CHECK: [[FLOAT:%[a-zA-Z0-9_]+]] = tvm_ffi.to [[FLOAT_ARG]] : f64 -> !tvm_ffi.float
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
     : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}

// -----

// CHECK-LABEL: func.func @is_exception(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.any
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = tvm_ffi.isa %[[INPUT]] is !tvm_ffi.exception
// CHECK: return %[[MATCH]]
func.func @is_exception(%input: !tvm_ffi.any) -> i1 {
  %match = tvm_ffi.isa %input is !tvm_ffi.exception : !tvm_ffi.any
  return %match : i1
}

// -----

// CHECK-LABEL: tvm_ffi.func private @private_function
tvm_ffi.func private @private_function() -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %value = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
