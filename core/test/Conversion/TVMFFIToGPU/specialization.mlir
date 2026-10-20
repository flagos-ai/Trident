//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file --inline --convert-tvm-ffi-to-gpu | FileCheck %s

// CHECK-LABEL: func.func @validate
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor
// CHECK-SAME: %[[SPECIALIZED:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// CHECK-NOT: call @launch
// CHECK: %[[GUARD_OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]]
// CHECK: %[[GUARD_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[GUARD_OBJECT]]
// CHECK: %[[GUARD_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[GUARD_TENSOR]]
// CHECK: llvm.ptrtoint %[[GUARD_DATA]]
// CHECK: llvm.urem
// CHECK: %[[TENSOR_CHECKS:[a-zA-Z0-9_]+]] = arith.andi
// CHECK: %[[EXPECTED:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[CONSTANT_CHECK:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[SPECIALIZED]], %[[EXPECTED]] : !tvm_ffi.int
// CHECK: %[[CHECKS_WITH_CONSTANT:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_CHECKS]], %[[CONSTANT_CHECK]]
// CHECK: %[[FLOAT_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[FLOAT:[a-zA-Z0-9_]+]]
// CHECK: %[[CAST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast
// CHECK: llvm.urem
// CHECK: llvm.icmp "eq"
// CHECK: %[[ALL_CHECKS:[a-zA-Z0-9_]+]] = arith.andi
// CHECK: scf.if %[[ALL_CHECKS]]
// CHECK: %[[LAUNCH_OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]]
// CHECK: %[[LAUNCH_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[LAUNCH_OBJECT]]
// CHECK: %[[LAUNCH_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[LAUNCH_TENSOR]]
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[LAUNCH_DATA]] : !llvm.ptr
// CHECK: tvm_ffi.monad.pure
// CHECK: scf.yield
// CHECK: } else {
// CHECK: %[[REJECTED:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[REJECTED]]
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch(%tensor: !tvm_ffi.tensor,
      %specialized: !tvm_ffi.int, %value: !tvm_ffi.float)
      -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !tvm_ffi.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %specialized : !tvm_ffi.int #tvm_ffi.constant_specialization<value = 1 : i64>, %value : !tvm_ffi.float #tvm_ffi.variable_specialization<kind = f32, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  func.func @validate(%tensor: !tvm_ffi.tensor, %value: !tvm_ffi.float,
      %specialized: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.any> {
    %result = func.call @launch(%tensor, %specialized, %value)
        : (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.float) -> !tvm_ffi.int
    %success = tvm_ffi.upcast %result : !tvm_ffi.int -> !tvm_ffi.any

    %returned = tvm_ffi.monad.pure %success : (!tvm_ffi.any) -> !tvm_ffi.monad.result<!tvm_ffi.any>
    return %returned : !tvm_ffi.monad.result<!tvm_ffi.any>
  }
}

// -----

// CHECK-LABEL: func.func @validate_scalar_constants
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]: !tvm_ffi.bool
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// CHECK-SAME: %[[SCALE:[a-zA-Z0-9_]+]]: !tvm_ffi.float
// CHECK: %[[INITIAL:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[TRUE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: %[[BOOL_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[FLAG]], %[[TRUE]] : !tvm_ffi.bool
// CHECK: %[[BOOL_CHECK:[a-zA-Z0-9_]+]] = arith.andi %[[INITIAL]], %[[BOOL_EQUAL]] : i1
// CHECK: %[[VALUE_NATIVE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[VALUE]]
// CHECK: %[[EXPECTED_FLOAT:[a-zA-Z0-9_.]+]] = tvm_ffi.constant.float 1.000000e+00
// CHECK: %[[FLOAT_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[SCALE]], %[[EXPECTED_FLOAT]] : !tvm_ffi.float
// CHECK: %[[SCALAR_CHECK:[a-zA-Z0-9_]+]] = arith.andi %[[BOOL_CHECK]], %[[FLOAT_EQUAL]] : i1
// CHECK: scf.if %[[SCALAR_CHECK]]
// CHECK: %[[LAUNCH_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[VALUE]]
// CHECK: %[[LAUNCH_I32:[a-zA-Z0-9_]+]] = arith.trunci %[[LAUNCH_VALUE]] : i64 to i32
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[LAUNCH_I32]] : i32
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch_scalar_constants(%flag: !tvm_ffi.bool,
      %value: !tvm_ffi.int, %scale: !tvm_ffi.float) -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    %launch_1 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%flag : !tvm_ffi.bool #tvm_ffi.constant_specialization<value = true>, %value : !tvm_ffi.int #tvm_ffi.variable_specialization<kind = i32>, %scale : !tvm_ffi.float #tvm_ffi.constant_specialization<value = 1.000000e+00 : f64>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  func.func @validate_scalar_constants(%flag: !tvm_ffi.bool,
      %value: !tvm_ffi.int, %scale: !tvm_ffi.float) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>> {
    %result = func.call @launch_scalar_constants(%flag, %value, %scale)
        : (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) -> !tvm_ffi.int
    %success = tvm_ffi.upcast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>

    %returned = tvm_ffi.monad.pure %success : (!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
    return %returned : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
  }
}

// -----

// CHECK-LABEL: func.func @validate_string
// CHECK-SAME: %[[ACTIVATION:[a-zA-Z0-9_]+]]: !tvm_ffi.raw_str
// CHECK: %[[INITIAL_STRING:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[EXPECTED_STRING:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "leaky_relu"
// CHECK: %[[STRING_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[ACTIVATION]], %[[EXPECTED_STRING]] : !tvm_ffi.raw_str
// CHECK: %[[STRING_CHECK:[a-zA-Z0-9_]+]] = arith.andi %[[INITIAL_STRING]], %[[STRING_EQUAL]] : i1
// CHECK: scf.if %[[STRING_CHECK]]
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[ZERO_STRING:[a-zA-Z0-9_]+]] : i64, %[[ZERO_STRING]] : i64)
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch_string(%activation: !tvm_ffi.raw_str)
      -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    %launch_2 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%activation : !tvm_ffi.raw_str #tvm_ffi.constant_specialization<value = "leaky_relu">) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  func.func @validate_string(%activation: !tvm_ffi.raw_str) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>> {
    %result = func.call @launch_string(%activation)
        : (!tvm_ffi.raw_str) -> !tvm_ffi.int
    %success = tvm_ffi.upcast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>

    %returned = tvm_ffi.monad.pure %success : (!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
    return %returned : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
  }
}

// -----

// CHECK-LABEL: func.func @validate_tuple
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK: %[[INITIAL_TUPLE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[TRUE_TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: %[[NAME:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "name"
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[INNER_HANDLE:[a-zA-Z0-9_]+]](%[[TRUE_TUPLE]], %[[NAME]])
// CHECK: %[[EXPECTED_TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[OUTER_HANDLE:[a-zA-Z0-9_]+]](%[[ONE]], %[[INNER]])
// CHECK: tvm_ffi.ObjectDecRef %[[OUTER_HANDLE]]
// CHECK: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK: %[[TUPLE_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[TUPLE]], %[[EXPECTED_TUPLE]] : !tvm_ffi.c.array
// CHECK: tvm_ffi.ObjectDecRef %[[EXPECTED_TUPLE]]
// CHECK: %[[TUPLE_CHECK:[a-zA-Z0-9_]+]] = arith.andi %[[INITIAL_TUPLE]], %[[TUPLE_EQUAL]] : i1
// CHECK: scf.if %[[TUPLE_CHECK]]
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[ZERO_TUPLE:[a-zA-Z0-9_]+]] : i64, %[[ZERO_TUPLE]] : i64)
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch_tuple(
      %value: !tvm_ffi.c.array) -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    %launch_3 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi.c.array #tvm_ffi.constant_specialization<value = [1 : i64, [true, "name"]]>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  func.func @validate_tuple(
      %value: !tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>> {
    %result = func.call @launch_tuple(%value)
        : (!tvm_ffi.c.array) -> !tvm_ffi.int
    %success = tvm_ffi.upcast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>

    %returned = tvm_ffi.monad.pure %success : (!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
    return %returned : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>>
  }
}
