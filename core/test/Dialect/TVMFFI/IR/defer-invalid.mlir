// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --verify-diagnostics

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
