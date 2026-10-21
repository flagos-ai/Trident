// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s --check-prefix=PARSE
// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s --check-prefix=FOLD

// Widening, membership/subset compatibility and function payloads, followed by
// representative identity parsing and folding.

// PARSE-LABEL: func.func @any_to_int(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.any -> !tvm_ffi.int
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.int
func.func @any_to_int(%input: !tvm_ffi.any) -> !tvm_ffi.int {
  %result = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// PARSE-LABEL: func.func @any_to_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
func.func @any_to_union(%input: !tvm_ffi.any) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
}

// PARSE-LABEL: func.func @downcast_function(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.any -> !tvm_ffi.c.function
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.c.function
func.func @downcast_function(%input: !tvm_ffi.any) -> !tvm_ffi.c.function {
  %result = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.c.function
  return %result : !tvm_ffi.c.function
}

// PARSE-LABEL: func.func @downcast_function_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>
func.func @downcast_function_union(%input: !tvm_ffi.any) -> !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none> {
  %result = tvm_ffi.downcast %input : !tvm_ffi.any -> !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>
  return %result : !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>
}

// PARSE-LABEL: func.func @union_to_any(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.any
func.func @union_to_any(%input: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.any {
  %result = tvm_ffi.upcast %input : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// PARSE-LABEL: func.func @union_to_int(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.int
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.int
func.func @union_to_int(%input: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.int {
  %result = tvm_ffi.downcast %input : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// PARSE-LABEL: func.func @union_to_subset(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[INPUT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
func.func @union_to_subset(%input: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.downcast %input : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
}

// PARSE-LABEL: func.func @union_to_superset(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
func.func @union_to_superset(%input: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> {
  %result = tvm_ffi.upcast %input : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>
}

// PARSE-LABEL: func.func @upcast_function(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.c.function -> !tvm_ffi.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.any
func.func @upcast_function(%input: !tvm_ffi.c.function) -> !tvm_ffi.any {
  %result = tvm_ffi.upcast %input : !tvm_ffi.c.function -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// PARSE-LABEL: func.func @upcast_function_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none> -> !tvm_ffi.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.any
func.func @upcast_function_union(%input: !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none>) -> !tvm_ffi.any {
  %result = tvm_ffi.upcast %input : !tvm_ffi.union<!tvm_ffi.c.function, !tvm_ffi.none> -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}

// PARSE-LABEL: func.func @value_to_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]] : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
func.func @value_to_union(%input: !tvm_ffi.int) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
  %result = tvm_ffi.upcast %input : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
  return %result : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
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
func.func @function(%arg: !tvm_ffi.c.function) -> !tvm_ffi.c.function {
  %wide = tvm_ffi.upcast %arg : !tvm_ffi.c.function -> !tvm_ffi.c.function
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.c.function -> !tvm_ffi.c.function
  return %narrow : !tvm_ffi.c.function
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
