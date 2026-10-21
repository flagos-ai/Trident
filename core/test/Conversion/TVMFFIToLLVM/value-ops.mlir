// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-llvm --verify-each | FileCheck %s

// -----

// CHECK-LABEL: func.func @dtype_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = llvm.mlir.constant(73730 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PACKED]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @dtype_constant() -> !tvm_ffi.dtype {
  %value = "tvm_ffi.constant.dtype"() <{value = [2, 32, 1]}>
     : () -> !tvm_ffi.dtype
  return %value : !tvm_ffi.dtype
}

// -----

// CHECK-LABEL: func.func @int_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-7 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @int_constant() -> !tvm_ffi.int {
  %value = "tvm_ffi.constant.int"() <{value = -7 : i64}> : () -> !tvm_ffi.int
  return %value : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @none_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @none_constant() -> !tvm_ffi.none {
  %value = "tvm_ffi.constant.none"() : () -> !tvm_ffi.none
  return %value : !tvm_ffi.none
}

// -----

// CHECK-LABEL: func.func @int_bool(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: ConvertUnrealizedIntCastOp
// CHECK-NOT: tvm_ffi.to
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = llvm.trunc %[[PAYLOAD]] : i64 to i1
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL]] : i1 to i64
// CHECK: %[[UNDEF_RESULT:[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[TYPE_INDEX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[TYPE_RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[TYPE_INDEX]], %[[UNDEF_RESULT]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[KIND_INDEX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[KIND_RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND_INDEX]], %[[TYPE_RESULT]][1] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[INT]], %[[KIND_RESULT]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @int_bool(%arg0: !tvm_ffi.bool) -> !tvm_ffi.int {
  %bool = tvm_ffi.get %arg0 : !tvm_ffi.bool -> i1
  %int = arith.extui %bool : i1 to i64
  %0 = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  return %0 : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @as_dlpack_tensor(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[PAYLOAD]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK-NEXT: return %[[TENSOR]] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
func.func @as_dlpack_tensor(%object: !tvm_ffi.object)
    -> !dlpack.dl_tensor {
  %tensor = tvm_ffi.as %object : !tvm_ffi.object -> !dlpack.dl_tensor
  return %tensor : !dlpack.dl_tensor
}

// Test the kernel-argument construction shape: !tvm_ffi.int -> i64 -> i32
// CHECK-LABEL: func.func @cast_int_to_i32(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK: %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[I32:[a-zA-Z0-9_]+]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK-NOT: tvm_ffi.get
// CHECK: return %[[I32]] : i32
func.func @cast_int_to_i32(%arg0: !tvm_ffi.int) -> i32 {
  %native = tvm_ffi.get %arg0 : !tvm_ffi.int -> i64
  %0 = llvm.trunc %native : i64 to i32
  return %0 : i32
}

// CHECK-LABEL: func.func @get_native_scalars(
// CHECK-SAME: %[[BOOL_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[INT_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[FLOAT_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK: %[[BOOL_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[BOOL_ARG]][2]
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = llvm.trunc %[[BOOL_PAYLOAD]] : i64 to i1
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[INT_ARG]][2]
// CHECK: %[[FLOAT_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[FLOAT_ARG]][2]
// CHECK: %[[FLOAT:[a-zA-Z0-9_]+]] = llvm.bitcast %[[FLOAT_PAYLOAD]] : i64 to f64
// CHECK: return %[[BOOL]], %[[INT]], %[[FLOAT]] : i1, i64, f64
func.func @get_native_scalars(%bool: !tvm_ffi.bool, %int: !tvm_ffi.int,
                              %float: !tvm_ffi.float) -> (i1, i64, f64) {
  %bool_value = tvm_ffi.get %bool : !tvm_ffi.bool -> i1
  %int_value = tvm_ffi.get %int : !tvm_ffi.int -> i64
  %float_value = tvm_ffi.get %float : !tvm_ffi.float -> f64
  return %bool_value, %int_value, %float_value : i1, i64, f64
}

// CHECK-LABEL: func.func @get_object(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT: return %[[OBJECT]] : !llvm.ptr
func.func @get_object(%tensor: !tvm_ffi.c.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %tensor : !tvm_ffi.c.tensor -> !tvm_ffi.object
  return %object : !tvm_ffi.object
}

// CHECK-LABEL: func.func @tensor_literal() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[CREATE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA:[a-zA-Z0-9_]+]], %[[DATA_COUNT:[a-zA-Z0-9_]+]], %[[SHAPE:[a-zA-Z0-9_]+]], %[[STRIDES:[a-zA-Z0-9_]+]], %[[DTYPE:[a-zA-Z0-9_]+]], %[[DEVICE:[a-zA-Z0-9_]+]], %[[REQUIRES_GRAD:[a-zA-Z0-9_]+]], %[[MEMORY_FORMAT:[a-zA-Z0-9_]+]], %[[OUT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[OUT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[EMPTY_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_empty_strided(%[[EMPTY_COUNT:[a-zA-Z0-9_]+]], %[[EMPTY_SHAPE:[a-zA-Z0-9_]+]], %[[EMPTY_STRIDES:[a-zA-Z0-9_]+]], %[[EMPTY_DTYPE:[a-zA-Z0-9_]+]], %[[EMPTY_DEVICE:[a-zA-Z0-9_]+]], %[[EMPTY_REQUIRES_GRAD:[a-zA-Z0-9_]+]], %[[EMPTY_OUT:[a-zA-Z0-9_]+]]) : (i64, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_OUT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[COPY_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_copy_(%[[EMPTY_TENSOR]], %[[TENSOR]], %[[NON_BLOCKING:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK: %[[DELETE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_delete_tensor_object(%[[TENSOR]]) : (!llvm.ptr) -> i32
// CHECK: %[[HANDLE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @tensor_literal() -> !tvm_ffi.c.tensor {
  %literal = tvm_ffi.tensor.literal dense<[1.0, 2.0]> : tensor<2xf32>
     : !tvm_ffi.c.tensor
  return %literal : !tvm_ffi.c.tensor
}

// CHECK-LABEL: func.func @to_native_scalars(
// CHECK-SAME: %[[BOOL_ARG:[a-zA-Z0-9_]+]]: i1, %[[INT_ARG:[a-zA-Z0-9_]+]]: i64, %[[FLOAT_ARG:[a-zA-Z0-9_]+]]: f64)
// CHECK: %[[BOOL_VALUE:[a-zA-Z0-9_]+]] = llvm.zext %[[BOOL_ARG]] : i1 to i64
// CHECK: %[[BOOL_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[INT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[FLOAT_VALUE:[a-zA-Z0-9_]+]] = llvm.bitcast %[[FLOAT_ARG]] : f64 to i64
// CHECK: %[[FLOAT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32)
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
     : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}
