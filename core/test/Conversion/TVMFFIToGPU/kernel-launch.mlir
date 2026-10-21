// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-gpu --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.exception' --implicit-check-not='tvm_ffi.return'

// Only kernel lowering runs here: semantic ABI values remain available.
// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK: gpu.binary @kernel
// CHECK-LABEL: func.func @test_kernel_launch(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.int
// CHECK: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]] : !tvm_ffi.c.tensor -> !tvm_ffi.object
// CHECK: %[[DLTENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[OBJECT]] : !tvm_ffi.object -> !dlpack.dl_tensor
// CHECK: %[[DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[DLTENSOR]] : !dlpack.dl_tensor -> !llvm.ptr
// CHECK: %[[NATIVE_SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR]] : !tvm_ffi.int -> i64
// CHECK: scf.if
// CHECK: %[[LAUNCH_OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[TENSOR]]
// CHECK: %[[LAUNCH_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[LAUNCH_OBJECT]]
// CHECK: %[[LAUNCH_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[LAUNCH_TENSOR]]
// CHECK: %[[LAUNCH_SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR]]
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DEVICE_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[ONE]] x i32
// CHECK: llvm.call @aoti_torch_get_current_device_index(%[[DEVICE_SLOT]]) : (!llvm.ptr) -> i32
// CHECK: %[[DEVICE:[a-zA-Z0-9_]+]] = llvm.load %[[DEVICE_SLOT]] : !llvm.ptr -> i32
// CHECK: %[[CUDA:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[STREAM:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIEnvGetStream(%[[CUDA]], %[[DEVICE]]) : (i32, i32) -> !llvm.ptr
// CHECK: gpu.launch_func <%[[STREAM]] : !llvm.ptr> @kernel::@entry
// CHECK: args(%[[LAUNCH_DATA]] : !llvm.ptr, %[[LAUNCH_SCALAR]] : i64,
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes { gpu.container_module } {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func @test_kernel_launch(
    %tensor: !tvm_ffi.c.tensor,
    %scalar: !tvm_ffi.int) {
    %c32 = llvm.mlir.constant(32 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c128 = llvm.mlir.constant(128 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %shmem = llvm.mlir.constant(16384 : i32) : i32

    %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%c32, %c16, %c1) : i64 threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %scalar : !tvm_ffi.int #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  // CHECK-LABEL: func.func @monadic_launch(
  // CHECK: %[[OUTCOME:[a-zA-Z0-9_]+]] = scf.if %[[READY:[a-zA-Z0-9_]+]] -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
  // CHECK: gpu.launch_func
  // CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
  // CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NONE]]
  // CHECK: scf.yield %[[SUCCESS]]
  // CHECK: } else {
  // CHECK: %[[REJECTION:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
  // CHECK: scf.yield %[[REJECTION]]
  // CHECK: return %[[OUTCOME]]
  func.func @monadic_launch(%tensor: !tvm_ffi.c.tensor, %scalar: i64) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch = tvm_ffi.kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr>, %scalar : i64 #tvm_ffi.variable_specialization<kind = i64, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return %launch : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}

// -----

// CHECK-LABEL: func.func @validate
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor
// CHECK-SAME: %[[SPECIALIZED:[a-zA-Z0-9_]+]]: !tvm_ffi.int
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

  func.func @validate(%tensor: !tvm_ffi.c.tensor,
      %specialized: !tvm_ffi.int, %value: !tvm_ffi.float)
      -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !tvm_ffi.c.tensor #tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %specialized : !tvm_ffi.int #tvm_ffi.constant_specialization<value = 1 : i64>, %value : !tvm_ffi.float #tvm_ffi.variable_specialization<kind = f32, divisibility = 16>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    return %launch_0 : !tvm_ffi.monad.result<!tvm_ffi.none>
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

  func.func @validate_scalar_constants(%flag: !tvm_ffi.bool,
      %value: !tvm_ffi.int, %scale: !tvm_ffi.float) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch_1 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%flag : !tvm_ffi.bool #tvm_ffi.constant_specialization<value = true>, %value : !tvm_ffi.int #tvm_ffi.variable_specialization<kind = i32>, %scale : !tvm_ffi.float #tvm_ffi.constant_specialization<value = 1.000000e+00 : f64>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    return %launch_1 : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}

// -----


// CHECK-LABEL: func.func @validate_string
// CHECK-SAME: %[[ACTIVATION:[a-zA-Z0-9_]+]]: !tvm_ffi.str
// CHECK: %[[INITIAL_STRING:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[EXPECTED_STRING:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "leaky_relu"
// CHECK: %[[STRING_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[ACTIVATION]], %[[EXPECTED_STRING]] : !tvm_ffi.str, !tvm_ffi.raw_str
// CHECK: %[[STRING_CHECK:[a-zA-Z0-9_]+]] = arith.andi %[[INITIAL_STRING]], %[[STRING_EQUAL]] : i1
// CHECK: scf.if %[[STRING_CHECK]]
// CHECK: gpu.launch_func
// CHECK-SAME: args(%[[ZERO_STRING:[a-zA-Z0-9_]+]] : i64, %[[ZERO_STRING]] : i64)
// CHECK-NOT: tvm_ffi.kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func @validate_string(%activation: !tvm_ffi.str)
      -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch_2 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%activation : !tvm_ffi.str #tvm_ffi.constant_specialization<value = "leaky_relu">) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    return %launch_2 : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}

// -----


// CHECK-LABEL: func.func @validate_tuple
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array
// CHECK: %[[INITIAL_TUPLE:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK: %[[TRUE_TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: %[[NAME:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "name"
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[INNER_HANDLE:[a-zA-Z0-9_]+]] : !tvm_ffi.c.function(%[[TRUE_TUPLE]], %[[NAME]])
// CHECK: %[[EXPECTED_TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[OUTER_HANDLE:[a-zA-Z0-9_]+]] : !tvm_ffi.c.function(%[[ONE]], %[[INNER]])
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

  func.func @validate_tuple(
      %value: !tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
    %one = arith.constant 1 : i64
    %launch_3 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi.c.array #tvm_ffi.constant_specialization<value = [1 : i64, [true, "name"]]>) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    return %launch_3 : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
}
