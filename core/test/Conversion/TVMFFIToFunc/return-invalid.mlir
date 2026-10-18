// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-tvm-ffi-to-func --verify-diagnostics
// RUN: not trident-core-opt %s --convert-tvm-ffi-to-func 2>&1 | FileCheck %s --check-prefix=MISSING

// MISSING: error: run lower-tvm-ffi-monad before converting tvm_ffi.return to the ABI

// expected-error@+1 {{ABI return payloads must have a TVM FFI ABI type}}
tvm_ffi.func @native_export() -> !tvm_ffi.monad.result<i64> attributes {emit_tvm_ffi_abi} {
  %value = arith.constant 7 : i64
  %result = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
  tvm_ffi.return %result : !tvm_ffi.monad.result<i64>
}
