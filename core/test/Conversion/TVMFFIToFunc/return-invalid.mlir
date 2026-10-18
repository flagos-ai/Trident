// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-to-func --verify-diagnostics

// Native handles can use internal output slots, but are not packed ABI values.
// expected-error@+1 {{ABI return payloads must have a tagged TVM FFI ABI type}}
tvm_ffi.func @function_export() -> !tvm_ffi.monad.result<!tvm_ffi.function> attributes {emit_tvm_ffi_abi} {
  %result = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.function>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.function>
}
// -----

// expected-error@+1 {{ABI return payloads must have a tagged TVM FFI ABI type}}
tvm_ffi.func @native_export() -> !tvm_ffi.monad.result<i64> attributes {emit_tvm_ffi_abi} {
  %value = arith.constant 7 : i64
  %result = tvm_ffi.monad.pure %value : (i64) -> !tvm_ffi.monad.result<i64>
  tvm_ffi.return %result : !tvm_ffi.monad.result<i64>
}
// -----

// Native handles can use internal output slots, but are not packed ABI values.
// expected-error@+1 {{ABI return payloads must have a tagged TVM FFI ABI type}}
tvm_ffi.func @object_export() -> !tvm_ffi.monad.result<!tvm_ffi.object> attributes {emit_tvm_ffi_abi} {
  %result = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.object>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.object>
}
