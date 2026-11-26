// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @array_create(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{elements must match the array element type}}
  %result = tvm_ffi_api.array.create(%value) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.bool>
  return
}

// -----

func.func @array_create_payload(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{result #0 must be An array whose elements are TVM FFI C ABI values, but got '!tvm_ffi_abi.int'}}
  %result = tvm_ffi_api.array.create(%value) : (!tvm_ffi_abi.int) -> !tvm_ffi_abi.int
  return
}

// -----

func.func @array_get(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>, %index: i64) {
  // expected-error@+1 {{result must match array element type}}
  %result = tvm_ffi_api.array.get %array[%index] : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.bool
  return
}

// -----

func.func @array_size(%array: !tvm_ffi_api.array<!tvm_ffi_abi.int>) {
  // expected-error@+1 {{expected tvm_ffi_abi.int}}
  %result = tvm_ffi_api.array.size %array : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_abi.bool
  return
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFIAPI_ArrayType parameter}}
func.func private @array_native(!tvm_ffi_api.array<i64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFIAPI_ArrayType parameter}}
func.func private @array_result(!tvm_ffi_api.array<!tvm_ffi_monad.result<!tvm_ffi_abi.int>>)

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @result(!tvm_ffi_api.tuple<!tvm_ffi_monad.result<!tvm_ffi_abi.int>>)

// -----

func.func @tuple_construct_count(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{elements must match the tuple types}}
  %result = tvm_ffi_api.tuple.construct(%value) : (!tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int, !tvm_ffi_abi.int>
  return
}

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @tuple_native(!tvm_ffi_api.tuple<!tvm_ffi_abi.int, i64>)

// -----

func.func @tuple_unconstruct_count(%tuple: !tvm_ffi_api.tuple<!tvm_ffi_abi.int>) {
  // expected-error@+1 {{results must match the tuple types}}
  tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_abi.int>) -> ()
  return
}

// -----

func.func @arguments() {
  // expected-error@+1 {{body must have no arguments}}
  tvm_ffi_abi.defer {
  ^bb0(%argument: i64):
    tvm_ffi_abi.defer_yield
  }
  return
}

// -----

func.func @missing_yield() {
  // expected-error@+1 {{body must end with tvm_ffi_abi.defer_yield}}
  tvm_ffi_abi.defer {
    func.return
  }
  return
}

// -----

func.func @nested() {
  // expected-error@+1 {{body must contain only operations without regions or terminators}}
  tvm_ffi_abi.defer {
    tvm_ffi_abi.defer {
      tvm_ffi_abi.defer_yield
    }
    tvm_ffi_abi.defer_yield
  }
  return
}

// -----

func.func @wrong_parent() {
  // expected-error@+1 {{must terminate tvm_ffi_abi.defer}}
  tvm_ffi_abi.defer_yield
}

// -----

module {
  func.func @invalid_constant_specialization_type(%value: !tvm_ffi_abi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{constant specialization requires an i1, i64, f64, string, or tuple value}}
    %result = "tvm_ffi_api.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [#tvm_ffi_api.constant_specialization<value = 1 : i32>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    func.return
  }
}

// -----

module {
  func.func @invalid_specialization_element(%value: !tvm_ffi_abi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi_api.kernel_launch' op attribute 'specializations' failed to satisfy constraint: array of Triton argument specializations}}
    %result = "tvm_ffi_api.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [42 : i64]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    func.return
  }
}

// -----

module {
  func.func @missing_specialization(%value: !tvm_ffi_abi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+4 {{expected attribute value}}
    %launch_0 = tvm_ffi_api.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    func.return
  }
}

// -----

module {
  func.func @specializations_size(%tensor: !tvm_ffi_abi.tensor, %value: !tvm_ffi_abi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi_api.kernel_launch' op specializations and kernel operands must have the same size}}
    %result = "tvm_ffi_api.kernel_launch"(%one, %one, %one, %one, %one, %one, %tensor, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2>, specializations = [#tvm_ffi_api.variable_specialization<kind = !llvm.ptr, divisibility = 16>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi_abi.tensor, !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_abi.none>
    func.return
  }
}

// -----
module {
  func.func @invalid_payload() {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{must return Result<None>}}
    %launch = tvm_ffi_api.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
        args (%one : i64 #tvm_ffi_api.variable_specialization<kind = i64>) -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
    func.return
  }
}

// -----

func.func @get_int_width(%arg: !tvm_ffi_abi.int) {
  // expected-error@+1 {{'tvm_ffi_api.get' op unsupported get from '!tvm_ffi_abi.int' to 'i128'}}
  %value = "tvm_ffi_api.get"(%arg) : (!tvm_ffi_abi.int) -> i128
  return
}

// -----

func.func @get_unsupported_array(%arg: !tvm_ffi_abi.array) {
  // expected-error@+1 {{'tvm_ffi_api.get' op unsupported get from '!tvm_ffi_abi.array' to 'i64'}}
  %value = "tvm_ffi_api.get"(%arg) : (!tvm_ffi_abi.array) -> i64
  return
}

// -----

func.func @invalid_dtype_constant() {
  // expected-error@+1 {{dtype result requires [code, bits, lanes]}}
  %value = "tvm_ffi_abi.constant.dtype"() <{value = [1, 2]}> : () -> !tvm_ffi_abi.dtype
  return
}

// -----

// -----

func.func @to_mismatched_native_type(%value: i64) {
  // expected-error@+1 {{'tvm_ffi_abi.to' op unsupported native-to-TVM FFI conversion from 'i64' to '!tvm_ffi_abi.bool'}}
  %result = tvm_ffi_abi.to %value : i64 -> !tvm_ffi_abi.bool
  return
}

// -----

// HasObject and Contains satisfy the combined constraint. Native Object is
// admitted separately, without acquiring C ABI object classification.
func.func @accepted(%any: !tvm_ffi_api.any, %array: !tvm_ffi_abi.array, %exception: !tvm_ffi_abi.exception, %function: !tvm_ffi_abi.function, %object: !tvm_ffi_abi.object, %str: !tvm_ffi_abi.str, %tensor: !tvm_ffi_abi.tensor, %union: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) {
  tvm_ffi_abi.object.inc_ref %any : !tvm_ffi_api.any
  tvm_ffi_abi.object.dec_ref %array : !tvm_ffi_abi.array
  tvm_ffi_abi.object.inc_ref %exception : !tvm_ffi_abi.exception
  tvm_ffi_abi.object.dec_ref %function : !tvm_ffi_abi.function
  tvm_ffi_abi.object.inc_ref %object : !tvm_ffi_abi.object
  tvm_ffi_abi.object.dec_ref %str : !tvm_ffi_abi.str
  tvm_ffi_abi.object.inc_ref %tensor : !tvm_ffi_abi.tensor
  tvm_ffi_abi.object.dec_ref %union : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  return
}

// -----

func.func @array(%value: !tvm_ffi_api.array<!tvm_ffi_abi.int>) {
  // expected-error@+1 {{operand #0 must be TVM FFI ABI reference}}
  tvm_ffi_abi.object.inc_ref %value : !tvm_ffi_api.array<!tvm_ffi_abi.int>
  return
}

// -----

func.func @result(%value: !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>) {
  // expected-error@+1 {{operand #0 must be TVM FFI ABI reference}}
  tvm_ffi_abi.object.dec_ref %value : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
  return
}

// -----

func.func @scalar(%value: !tvm_ffi_abi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI ABI reference}}
  tvm_ffi_abi.object.inc_ref %value : !tvm_ffi_abi.int
  return
}

// -----

func.func @is_unindexed(%input: !tvm_ffi_api.any) -> i1 {
  // expected-error@+1 {{target must have a concrete runtime type index}}
  %match = tvm_ffi_api.isa %input is !tvm_ffi_api.str : !tvm_ffi_api.any
  return %match : i1
}

// -----

func.func @native_launch_rejects_semantic_operand(%value: !tvm_ffi_abi.int) {
  %one = arith.constant 1 : i64
  // expected-error@+1 {{kernel operands must have native integer, float or pointer types}}
  tvm_ffi_abi.native_kernel_launch @kernel::@entry
    blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
    args (%value : !tvm_ffi_abi.int)
  return
}

// -----

func.func @global_native_argument(%value: i64) {
  // expected-error@+1 {{operand #0 must be variadic of TVM FFI semantic value}}
  %result = tvm_ffi_api.global.call "test.identity"(%value) : (i64) -> !tvm_ffi_abi.int
  return
}

// -----

func.func @global_result_result() {
  // expected-error@+1 {{result #0 must be TVM FFI semantic value}}
  %result = tvm_ffi_api.global.call "test.identity"() : () -> !tvm_ffi_monad.result<!tvm_ffi_abi.int>
  return
}
