// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: func.func private @type_array(
// CHECK-SAME: %[[VALUE_ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array) -> !tvm_ffi_abi.array
// CHECK: return %[[VALUE_ARRAY]] : !tvm_ffi_abi.array
func.func private @type_array(%value: !tvm_ffi_abi.array) -> !tvm_ffi_abi.array {
  return %value : !tvm_ffi_abi.array
}

// CHECK-LABEL: func.func private @type_bool(
// CHECK-SAME: %[[VALUE_BOOL:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.bool) -> !tvm_ffi_abi.bool
// CHECK: return %[[VALUE_BOOL]] : !tvm_ffi_abi.bool
func.func private @type_bool(%value: !tvm_ffi_abi.bool) -> !tvm_ffi_abi.bool {
  return %value : !tvm_ffi_abi.bool
}

// CHECK-LABEL: func.func private @type_c_any(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any
// CHECK: return %[[VALUE]] : !tvm_ffi_abi.any
func.func private @type_c_any(%value: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any {
  return %value : !tvm_ffi_abi.any
}

// CHECK-LABEL: func.func private @type_device(
// CHECK-SAME: %[[VALUE_DEVICE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.device) -> !tvm_ffi_abi.device
// CHECK: return %[[VALUE_DEVICE]] : !tvm_ffi_abi.device
func.func private @type_device(%value: !tvm_ffi_abi.device) -> !tvm_ffi_abi.device {
  return %value : !tvm_ffi_abi.device
}

// CHECK-LABEL: func.func private @type_dtype(
// CHECK-SAME: %[[VALUE_DTYPE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.dtype) -> !tvm_ffi_abi.dtype
// CHECK: return %[[VALUE_DTYPE]] : !tvm_ffi_abi.dtype
func.func private @type_dtype(%value: !tvm_ffi_abi.dtype) -> !tvm_ffi_abi.dtype {
  return %value : !tvm_ffi_abi.dtype
}

// CHECK-LABEL: func.func private @type_float(
// CHECK-SAME: %[[VALUE_FLOAT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.float) -> !tvm_ffi_abi.float
// CHECK: return %[[VALUE_FLOAT]] : !tvm_ffi_abi.float
func.func private @type_float(%value: !tvm_ffi_abi.float) -> !tvm_ffi_abi.float {
  return %value : !tvm_ffi_abi.float
}

// CHECK-LABEL: func.func private @type_function(
// CHECK-SAME: %[[VALUE_FUNCTION:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function) -> !tvm_ffi_abi.function
// CHECK: return %[[VALUE_FUNCTION]] : !tvm_ffi_abi.function
func.func private @type_function(%value: !tvm_ffi_abi.function) -> !tvm_ffi_abi.function {
  return %value : !tvm_ffi_abi.function
}

// CHECK-LABEL: func.func private @type_int(
// CHECK-SAME: %[[VALUE_INT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int
// CHECK: return %[[VALUE_INT]] : !tvm_ffi_abi.int
func.func private @type_int(%value: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  return %value : !tvm_ffi_abi.int
}

// CHECK-LABEL: func.func private @type_none(
// CHECK-SAME: %[[VALUE_NONE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.none) -> !tvm_ffi_abi.none
// CHECK: return %[[VALUE_NONE]] : !tvm_ffi_abi.none
func.func private @type_none(%value: !tvm_ffi_abi.none) -> !tvm_ffi_abi.none {
  return %value : !tvm_ffi_abi.none
}

// CHECK-LABEL: func.func private @type_object(
// CHECK-SAME: %[[VALUE_OBJECT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.object) -> !tvm_ffi_abi.object
// CHECK: return %[[VALUE_OBJECT]] : !tvm_ffi_abi.object
func.func private @type_object(%value: !tvm_ffi_abi.object) -> !tvm_ffi_abi.object {
  return %value : !tvm_ffi_abi.object
}

// CHECK-LABEL: func.func private @type_raw_str(
// CHECK-SAME: %[[VALUE_RAW_STR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.raw_str) -> !tvm_ffi_abi.raw_str
// CHECK: return %[[VALUE_RAW_STR]] : !tvm_ffi_abi.raw_str
func.func private @type_raw_str(%value: !tvm_ffi_abi.raw_str) -> !tvm_ffi_abi.raw_str {
  return %value : !tvm_ffi_abi.raw_str
}

// CHECK-LABEL: func.func private @type_small_str(
// CHECK-SAME: %[[VALUE_SMALL_STR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.small_str) -> !tvm_ffi_abi.small_str
// CHECK: return %[[VALUE_SMALL_STR]] : !tvm_ffi_abi.small_str
func.func private @type_small_str(%value: !tvm_ffi_abi.small_str) -> !tvm_ffi_abi.small_str {
  return %value : !tvm_ffi_abi.small_str
}

// CHECK-LABEL: func.func private @type_str(
// CHECK-SAME: %[[VALUE_STR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.str) -> !tvm_ffi_abi.str
// CHECK: return %[[VALUE_STR]] : !tvm_ffi_abi.str
func.func private @type_str(%value: !tvm_ffi_abi.str) -> !tvm_ffi_abi.str {
  return %value : !tvm_ffi_abi.str
}

// CHECK-LABEL: func.func private @type_tensor(
// CHECK-SAME: %[[VALUE_TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor
// CHECK: return %[[VALUE_TENSOR]] : !tvm_ffi_abi.tensor
func.func private @type_tensor(%value: !tvm_ffi_abi.tensor) -> !tvm_ffi_abi.tensor {
  return %value : !tvm_ffi_abi.tensor
}
