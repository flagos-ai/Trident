// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: func.func private @type_any(
// CHECK-SAME: %[[VALUE_ANY:[a-zA-Z0-9_]+]]: !tvm_ffi.any) -> !tvm_ffi.any
// CHECK: return %[[VALUE_ANY]] : !tvm_ffi.any
func.func private @type_any(%value: !tvm_ffi.any) -> !tvm_ffi.any {
  return %value : !tvm_ffi.any
}

// CHECK-LABEL: func.func private @type_array(
// CHECK-SAME: %[[VALUE_ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array) -> !tvm_ffi.c.array
// CHECK: return %[[VALUE_ARRAY]] : !tvm_ffi.c.array
func.func private @type_array(%value: !tvm_ffi.c.array) -> !tvm_ffi.c.array {
  return %value : !tvm_ffi.c.array
}

// CHECK-LABEL: func.func private @type_bool(
// CHECK-SAME: %[[VALUE_BOOL:[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> !tvm_ffi.bool
// CHECK: return %[[VALUE_BOOL]] : !tvm_ffi.bool
func.func private @type_bool(%value: !tvm_ffi.bool) -> !tvm_ffi.bool {
  return %value : !tvm_ffi.bool
}

// CHECK-LABEL: func.func private @type_c_any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK: return %[[VALUE]] : !tvm_ffi.c.any
func.func private @type_c_any(%value: !tvm_ffi.c.any) -> !tvm_ffi.c.any {
  return %value : !tvm_ffi.c.any
}

// CHECK-LABEL: func.func private @type_device(
// CHECK-SAME: %[[VALUE_DEVICE:[a-zA-Z0-9_]+]]: !tvm_ffi.device) -> !tvm_ffi.device
// CHECK: return %[[VALUE_DEVICE]] : !tvm_ffi.device
func.func private @type_device(%value: !tvm_ffi.device) -> !tvm_ffi.device {
  return %value : !tvm_ffi.device
}

// CHECK-LABEL: func.func private @type_dtype(
// CHECK-SAME: %[[VALUE_DTYPE:[a-zA-Z0-9_]+]]: !tvm_ffi.dtype) -> !tvm_ffi.dtype
// CHECK: return %[[VALUE_DTYPE]] : !tvm_ffi.dtype
func.func private @type_dtype(%value: !tvm_ffi.dtype) -> !tvm_ffi.dtype {
  return %value : !tvm_ffi.dtype
}

// CHECK-LABEL: func.func private @type_extensions(
// CHECK-SAME: %[[EXTENSION:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>>
// CHECK: return %[[EXTENSION]] : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>>
func.func private @type_extensions(%value: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>>) -> !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>> {
  return %value : !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.c.exception>>
}
// CHECK-LABEL: func.func private @type_float(
// CHECK-SAME: %[[VALUE_FLOAT:[a-zA-Z0-9_]+]]: !tvm_ffi.float) -> !tvm_ffi.float
// CHECK: return %[[VALUE_FLOAT]] : !tvm_ffi.float
func.func private @type_float(%value: !tvm_ffi.float) -> !tvm_ffi.float {
  return %value : !tvm_ffi.float
}

// CHECK-LABEL: func.func private @type_function(
// CHECK-SAME: %[[VALUE_FUNCTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function) -> !tvm_ffi.c.function
// CHECK: return %[[VALUE_FUNCTION]] : !tvm_ffi.c.function
func.func private @type_function(%value: !tvm_ffi.c.function) -> !tvm_ffi.c.function {
  return %value : !tvm_ffi.c.function
}

// CHECK-LABEL: func.func private @type_int(
// CHECK-SAME: %[[VALUE_INT:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: return %[[VALUE_INT]] : !tvm_ffi.int
func.func private @type_int(%value: !tvm_ffi.int) -> !tvm_ffi.int {
  return %value : !tvm_ffi.int
}

// CHECK-LABEL: func.func private @type_none(
// CHECK-SAME: %[[VALUE_NONE:[a-zA-Z0-9_]+]]: !tvm_ffi.none) -> !tvm_ffi.none
// CHECK: return %[[VALUE_NONE]] : !tvm_ffi.none
func.func private @type_none(%value: !tvm_ffi.none) -> !tvm_ffi.none {
  return %value : !tvm_ffi.none
}

// CHECK-LABEL: func.func private @type_object(
// CHECK-SAME: %[[VALUE_OBJECT:[a-zA-Z0-9_]+]]: !tvm_ffi.object) -> !tvm_ffi.object
// CHECK: return %[[VALUE_OBJECT]] : !tvm_ffi.object
func.func private @type_object(%value: !tvm_ffi.object) -> !tvm_ffi.object {
  return %value : !tvm_ffi.object
}

// CHECK-LABEL: func.func private @type_raw_str(
// CHECK-SAME: %[[VALUE_RAW_STR:[a-zA-Z0-9_]+]]: !tvm_ffi.raw_str) -> !tvm_ffi.raw_str
// CHECK: return %[[VALUE_RAW_STR]] : !tvm_ffi.raw_str
func.func private @type_raw_str(%value: !tvm_ffi.raw_str) -> !tvm_ffi.raw_str {
  return %value : !tvm_ffi.raw_str
}

// CHECK-LABEL: func.func private @type_small_str(
// CHECK-SAME: %[[VALUE_SMALL_STR:[a-zA-Z0-9_]+]]: !tvm_ffi.small_str) -> !tvm_ffi.small_str
// CHECK: return %[[VALUE_SMALL_STR]] : !tvm_ffi.small_str
func.func private @type_small_str(%value: !tvm_ffi.small_str) -> !tvm_ffi.small_str {
  return %value : !tvm_ffi.small_str
}

// CHECK-LABEL: func.func private @type_str(
// CHECK-SAME: %[[VALUE_STR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.str) -> !tvm_ffi.c.str
// CHECK: return %[[VALUE_STR]] : !tvm_ffi.c.str
func.func private @type_str(%value: !tvm_ffi.c.str) -> !tvm_ffi.c.str {
  return %value : !tvm_ffi.c.str
}

// CHECK-LABEL: func.func private @type_tensor(
// CHECK-SAME: %[[VALUE_TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor
// CHECK: return %[[VALUE_TENSOR]] : !tvm_ffi.c.tensor
func.func private @type_tensor(%value: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor {
  return %value : !tvm_ffi.c.tensor
}
