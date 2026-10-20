// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: trident-core-opt %s --canonicalize | FileCheck %s --check-prefix=FOLD

// PARSE-LABEL: func.func @any(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @any(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @any(%arg: !tvm_ffi.any) -> !tvm_ffi.any {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.any -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.any
  return %narrow : !tvm_ffi.any
}

// PARSE-LABEL: func.func @array(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @array(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @array(%arg: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.array<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.array<!tvm_ffi.int>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.array<!tvm_ffi.int>
  return %narrow : !tvm_ffi.array<!tvm_ffi.int>
}

// PARSE-LABEL: func.func @function(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @function(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @function(%arg: !tvm_ffi.function) -> !tvm_ffi.function {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.function -> !tvm_ffi.function
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.function -> !tvm_ffi.function
  return %narrow : !tvm_ffi.function
}

// PARSE-LABEL: func.func @int(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @int(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @int(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.int -> !tvm_ffi.int
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.int -> !tvm_ffi.int
  return %narrow : !tvm_ffi.int
}

// PARSE-LABEL: func.func @tuple(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @tuple(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @tuple(%arg: !tvm_ffi.tuple<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.tuple<!tvm_ffi.int> -> !tvm_ffi.tuple<!tvm_ffi.int>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.tuple<!tvm_ffi.int> -> !tvm_ffi.tuple<!tvm_ffi.int>
  return %narrow : !tvm_ffi.tuple<!tvm_ffi.int>
}

// PARSE-LABEL: func.func @union(
// PARSE: tvm_ffi.upcast
// PARSE: tvm_ffi.downcast
// FOLD-LABEL: func.func @union(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @union(%arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %narrow : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

