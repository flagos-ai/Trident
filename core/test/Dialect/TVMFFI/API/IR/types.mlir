// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s

// CHECK-LABEL: func.func private @type_any(
// CHECK-SAME: %[[VALUE_ANY:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any) -> !tvm_ffi_api.any
// CHECK: return %[[VALUE_ANY]] : !tvm_ffi_api.any
func.func private @type_any(%value: !tvm_ffi_api.any) -> !tvm_ffi_api.any {
  return %value : !tvm_ffi_api.any
}

// CHECK-LABEL: func.func private @type_extensions(
// CHECK-SAME: %[[EXTENSION:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>>) -> !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>>
// CHECK: return %[[EXTENSION]] : !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>>
func.func private @type_extensions(%value: !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>>) -> !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>> {
  return %value : !tvm_ffi_monad.result<!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.exception>>
}
