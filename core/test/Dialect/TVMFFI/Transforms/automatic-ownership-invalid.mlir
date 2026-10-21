// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --allow-unregistered-dialect --decompose-tvm-ffi --split-input-file --verify-diagnostics

// Base references remain manually managed, including across CFG edges.
func.func @base_multiple_blocks(%input: !tvm_ffi.c.tensor, %condition: i1) -> !tvm_ffi.c.tensor {
  cf.cond_br %condition, ^first, ^second
^first:
  return %input : !tvm_ffi.c.tensor
^second:
  return %input : !tvm_ffi.c.tensor
}

// -----

func.func @loop_carried(%initial: !tvm_ffi.array<!tvm_ffi.int>, %bound: index) -> !tvm_ffi.array<!tvm_ffi.int> {
  %lower = arith.constant 0 : index
  %step = arith.constant 1 : index
  // expected-error@+1 {{unsupported region operation for automatic TVM FFI ownership}}
  %result = scf.for %iv = %lower to %bound step %step iter_args(%current = %initial) -> (!tvm_ffi.array<!tvm_ffi.int>) {
    scf.yield %current : !tvm_ffi.array<!tvm_ffi.int>
  }
  return %result : !tvm_ffi.array<!tvm_ffi.int>
}

// -----

func.func @loop_capture(%input: !tvm_ffi.tensor, %bound: index) {
  %lower = arith.constant 0 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lower to %bound step %step {
    // expected-error@+1 {{unsupported region operation for automatic TVM FFI ownership}}
    %rank = tvm_ffi.tensor.dim %input : !tvm_ffi.tensor
  }
  return
}

// -----

// expected-error@+1 {{automatic TVM FFI ownership requires a single-block region}}
func.func @multiple_blocks(%input: !tvm_ffi.tensor, %condition: i1) -> !tvm_ffi.tensor {
  cf.cond_br %condition, ^first, ^second
^first:
  return %input : !tvm_ffi.tensor
^second:
  return %input : !tvm_ffi.tensor
}

// -----

func.func @unsupported(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{failed to legalize operation 'test.container'}}
  "test.container"(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> ()
  return
}
