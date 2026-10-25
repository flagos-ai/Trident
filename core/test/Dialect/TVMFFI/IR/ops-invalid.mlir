// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

func.func @array_create(%value: !tvm_ffi.int) {
  // expected-error@+1 {{elements must match the array element type}}
  %result = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.bool>
  return
}

// -----

func.func @array_create_payload(%value: !tvm_ffi.int) {
  // expected-error@+1 {{result #0 must be An array whose elements are TVM FFI C ABI values, but got '!tvm_ffi.int'}}
  %result = tvm_ffi.array.create(%value) : (!tvm_ffi.int) -> !tvm_ffi.int
  return
}

// -----

func.func @array_get(%array: !tvm_ffi.array<!tvm_ffi.int>, %index: i64) {
  // expected-error@+1 {{result must match array element type}}
  %result = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.bool
  return
}

// -----

func.func @array_size(%array: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{expected tvm_ffi.int}}
  %result = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.bool
  return
}

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ArrayType parameter}}
func.func private @array_native(!tvm_ffi.array<i64>)

// -----

// expected-error@+2 {{invalid kind of type specified}}
// expected-error@+1 {{failed to parse TVMFFI_ArrayType parameter}}
func.func private @array_result(!tvm_ffi.array<!tvm_ffi.monad.result<!tvm_ffi.int>>)

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @result(!tvm_ffi.tuple<!tvm_ffi.monad.result<!tvm_ffi.int>>)

// -----

func.func @tuple_construct_count(%value: !tvm_ffi.int) {
  // expected-error@+1 {{elements must match the tuple types}}
  %result = tvm_ffi.tuple.construct(%value) : (!tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.int, !tvm_ffi.int>
  return
}

// -----

// expected-error@+1 {{tuple elements must be TVM FFI C ABI types}}
func.func private @tuple_native(!tvm_ffi.tuple<!tvm_ffi.int, i64>)

// -----

func.func @tuple_unconstruct_count(%tuple: !tvm_ffi.tuple<!tvm_ffi.int>) {
  // expected-error@+1 {{results must match the tuple types}}
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.int>) -> ()
  return
}

// -----

func.func @arguments() {
  // expected-error@+1 {{body must have no arguments}}
  tvm_ffi.defer {
  ^bb0(%argument: i64):
    tvm_ffi.defer_yield
  }
  return
}

// -----

func.func @missing_yield() {
  // expected-error@+1 {{body must end with tvm_ffi.defer_yield}}
  tvm_ffi.defer {
    func.return
  }
  return
}

// -----

func.func @nested() {
  // expected-error@+1 {{body must contain only operations without regions or terminators}}
  tvm_ffi.defer {
    tvm_ffi.defer {
      tvm_ffi.defer_yield
    }
    tvm_ffi.defer_yield
  }
  return
}

// -----

func.func @wrong_parent() {
  // expected-error@+1 {{must terminate tvm_ffi.defer}}
  tvm_ffi.defer_yield
}

// -----

module {
  func.func @invalid_constant_specialization_type(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{constant specialization requires an i1, i64, f64, string, or tuple value}}
    %result = "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [#tvm_ffi.constant_specialization<value = 1 : i32>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}

// -----

module {
  func.func @invalid_specialization_element(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi.kernel_launch' op attribute 'specializations' failed to satisfy constraint: array of Triton argument specializations}}
    %result = "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [42 : i64]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}

// -----

module {
  func.func @missing_specialization(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+4 {{expected attribute value}}
    %launch_0 = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}

// -----

module {
  func.func @specializations_size(%tensor: !tvm_ffi.c.tensor, %value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi.kernel_launch' op specializations and kernel operands must have the same size}}
    %result = "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %tensor, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2>, specializations = [#tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.c.tensor, !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    func.return
  }
}

// -----
module {
  func.func @invalid_payload() {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{must return Result<None>}}
    %launch = tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
        args (%one : i64 #tvm_ffi.variable_specialization<kind = i64>) -> !tvm_ffi.monad.result<!tvm_ffi.int>
    func.return
  }
}

// -----

func.func @get_int_width(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.int' to 'i128'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.int) -> i128
  return
}

// -----

func.func @get_unsupported_array(%arg: !tvm_ffi.c.array) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.c.array' to 'i64'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.c.array) -> i64
  return
}

// -----

func.func @invalid_dtype_constant() {
  // expected-error@+1 {{dtype result requires [code, bits, lanes]}}
  %value = "tvm_ffi.constant.dtype"() <{value = [1, 2]}> : () -> !tvm_ffi.dtype
  return
}

// -----

// -----

func.func @to_mismatched_native_type(%value: i64) {
  // expected-error@+1 {{'tvm_ffi.to' op unsupported native-to-TVM FFI conversion from 'i64' to '!tvm_ffi.bool'}}
  %result = tvm_ffi.to %value : i64 -> !tvm_ffi.bool
  return
}

// -----

// HasObject and Contains satisfy the combined constraint. Native Object is
// admitted separately, without acquiring C ABI object classification.
func.func @accepted(%any: !tvm_ffi.any, %array: !tvm_ffi.c.array, %exception: !tvm_ffi.c.exception, %function: !tvm_ffi.c.function, %object: !tvm_ffi.object, %str: !tvm_ffi.c.str, %tensor: !tvm_ffi.c.tensor, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  tvm_ffi.ObjectIncRef %any : !tvm_ffi.any
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.c.array
  tvm_ffi.ObjectIncRef %exception : !tvm_ffi.c.exception
  tvm_ffi.ObjectDecRef %function : !tvm_ffi.c.function
  tvm_ffi.ObjectIncRef %object : !tvm_ffi.object
  tvm_ffi.ObjectDecRef %str : !tvm_ffi.c.str
  tvm_ffi.ObjectIncRef %tensor : !tvm_ffi.c.tensor
  tvm_ffi.ObjectDecRef %union : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return
}

// -----

func.func @array(%value: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.array<!tvm_ffi.int>
  return
}

// -----

func.func @result(%value: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  return
}

// -----

func.func @scalar(%value: !tvm_ffi.int) {
  // expected-error@+1 {{operand #0 must be TVM FFI object handle}}
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.int
  return
}

// -----

func.func @is_unindexed(%input: !tvm_ffi.any) -> i1 {
  // expected-error@+1 {{target must have a concrete runtime type index}}
  %match = tvm_ffi.isa %input is !tvm_ffi.str : !tvm_ffi.any
  return %match : i1
}

// -----

func.func @native_launch_rejects_semantic_operand(%value: !tvm_ffi.int) {
  %one = arith.constant 1 : i64
  // expected-error@+1 {{kernel operands must have native integer, float or pointer types}}
  tvm_ffi.native_kernel_launch @kernel::@entry
    blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
    args (%value : !tvm_ffi.int)
  return
}
