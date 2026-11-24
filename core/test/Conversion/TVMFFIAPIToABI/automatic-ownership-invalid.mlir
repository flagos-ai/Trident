// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --allow-unregistered-dialect --convert-tvm-ffi-api-to-abi --split-input-file --verify-diagnostics

// Base references remain manually managed, including across CFG edges.
func.func @base_multiple_blocks(%input: !tvm_ffi_abi.tensor, %condition: i1) -> !tvm_ffi_abi.tensor {
  cf.cond_br %condition, ^first, ^second
^first:
  return %input : !tvm_ffi_abi.tensor
^second:
  return %input : !tvm_ffi_abi.tensor
}

// -----

// Region boundaries without semantic types remain manually managed.
func.func @base_regions(%input: !tvm_ffi_abi.tensor, %index: index) -> !tvm_ffi_abi.tensor {
  %value = scf.execute_region -> !tvm_ffi_abi.tensor {
    scf.yield %input : !tvm_ffi_abi.tensor
  }
  %selected = scf.index_switch %index -> !tvm_ffi_abi.tensor
  case 0 {
    scf.yield %value : !tvm_ffi_abi.tensor
  }
  default {
    scf.yield %input : !tvm_ffi_abi.tensor
  }
  return %selected : !tvm_ffi_abi.tensor
}

// -----

func.func @switch_capture(%index: index, %input: !tvm_ffi_api.tensor) {
  // expected-error@+1 {{failed to legalize operation 'scf.index_switch'}}
  %result = scf.index_switch %index -> !tvm_ffi_api.tensor
  case 0 {
    scf.yield %input : !tvm_ffi_api.tensor
  }
  default {
    %clone = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
    scf.yield %clone : !tvm_ffi_api.tensor
  }
  return
}

// -----

func.func @unsupported(%input: !tvm_ffi_api.array<!tvm_ffi_abi.int>) {
  // expected-error@+1 {{failed to legalize operation 'test.container'}}
  "test.container"(%input) : (!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> ()
  return
}
