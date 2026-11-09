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

// Region boundaries without semantic types remain manually managed.
func.func @base_regions(%input: !tvm_ffi.c.tensor, %index: index) -> !tvm_ffi.c.tensor {
  %value = scf.execute_region -> !tvm_ffi.c.tensor {
    scf.yield %input : !tvm_ffi.c.tensor
  }
  %selected = scf.index_switch %index -> !tvm_ffi.c.tensor
  case 0 {
    scf.yield %value : !tvm_ffi.c.tensor
  }
  default {
    scf.yield %input : !tvm_ffi.c.tensor
  }
  return %selected : !tvm_ffi.c.tensor
}

// -----

func.func @execute_cfg(%condition: i1, %input: !tvm_ffi.tensor) {
  // expected-error@+1 {{failed to legalize operation 'scf.execute_region'}}
  %result = scf.execute_region -> !tvm_ffi.tensor {
    %clone = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
    cf.cond_br %condition, ^left(%clone : !tvm_ffi.tensor), ^right
  ^left(%borrowed: !tvm_ffi.tensor):
    scf.yield %borrowed : !tvm_ffi.tensor
  ^right:
    scf.yield %clone : !tvm_ffi.tensor
  }
  return
}

// -----

func.func @switch_capture(%index: index, %input: !tvm_ffi.tensor) {
  // expected-error@+1 {{failed to legalize operation 'scf.index_switch'}}
  %result = scf.index_switch %index -> !tvm_ffi.tensor
  case 0 {
    scf.yield %input : !tvm_ffi.tensor
  }
  default {
    %clone = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
    scf.yield %clone : !tvm_ffi.tensor
  }
  return
}

// -----

func.func @unsupported(%input: !tvm_ffi.array<!tvm_ffi.int>) {
  // expected-error@+1 {{failed to legalize operation 'test.container'}}
  "test.container"(%input) : (!tvm_ffi.array<!tvm_ffi.int>) -> ()
  return
}
