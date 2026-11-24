// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | trident-core-opt | FileCheck %s --check-prefix=PARSE
// RUN: trident-core-opt %s --canonicalize --verify-each | FileCheck %s --check-prefix=FOLD

// Widening, membership/subset compatibility and function payloads, followed by
// representative identity parsing and folding.

// PARSE-LABEL: func.func @any_to_int(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_abi.int
func.func @any_to_int(%input: !tvm_ffi_api.any) -> !tvm_ffi_abi.int {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// PARSE-LABEL: func.func @any_to_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
func.func @any_to_union(%input: !tvm_ffi_api.any) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
}

// PARSE-LABEL: func.func @downcast_function(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.any -> !tvm_ffi_abi.function
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_abi.function
func.func @downcast_function(%input: !tvm_ffi_api.any) -> !tvm_ffi_abi.function {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_abi.function
  return %result : !tvm_ffi_abi.function
}

// PARSE-LABEL: func.func @downcast_function_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>
func.func @downcast_function_union(%input: !tvm_ffi_api.any) -> !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none> {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.any -> !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>
}

// PARSE-LABEL: func.func @union_to_any(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[INPUT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_api.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.any
func.func @union_to_any(%input: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>) -> !tvm_ffi_api.any {
  %result = tvm_ffi_api.upcast %input : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_api.any
  return %result : !tvm_ffi_api.any
}

// PARSE-LABEL: func.func @union_to_int(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_abi.int
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_abi.int
func.func @union_to_int(%input: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>) -> !tvm_ffi_abi.int {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// PARSE-LABEL: func.func @union_to_subset(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[INPUT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
func.func @union_to_subset(%input: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> {
  %result = tvm_ffi_api.downcast %input : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
}

// PARSE-LABEL: func.func @union_to_superset(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[INPUT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>
func.func @union_to_superset(%input: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none> {
  %result = tvm_ffi_api.upcast %input : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>
}

// PARSE-LABEL: func.func @upcast_function(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[INPUT]] : !tvm_ffi_abi.function -> !tvm_ffi_api.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.any
func.func @upcast_function(%input: !tvm_ffi_abi.function) -> !tvm_ffi_api.any {
  %result = tvm_ffi_api.upcast %input : !tvm_ffi_abi.function -> !tvm_ffi_api.any
  return %result : !tvm_ffi_api.any
}

// PARSE-LABEL: func.func @upcast_function_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[INPUT]] : !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none> -> !tvm_ffi_api.any
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.any
func.func @upcast_function_union(%input: !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none>) -> !tvm_ffi_api.any {
  %result = tvm_ffi_api.upcast %input : !tvm_ffi_api.union<!tvm_ffi_abi.function, !tvm_ffi_abi.none> -> !tvm_ffi_api.any
  return %result : !tvm_ffi_api.any
}

// PARSE-LABEL: func.func @value_to_union(
// PARSE-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// PARSE-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast %[[INPUT]] : !tvm_ffi_abi.int -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
// PARSE-NEXT: return %[[RESULT]] : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
func.func @value_to_union(%input: !tvm_ffi_abi.int) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> {
  %result = tvm_ffi_api.upcast %input : !tvm_ffi_abi.int -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
  return %result : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float>
}

// PARSE-LABEL: func.func @array(
// PARSE: tvm_ffi_api.upcast
// PARSE: tvm_ffi_api.downcast
// FOLD-LABEL: func.func @array(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @array(%arg: !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_api.array<!tvm_ffi_abi.int> {
  %wide = tvm_ffi_api.upcast %arg : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_api.array<!tvm_ffi_abi.int> -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  return %narrow : !tvm_ffi_api.array<!tvm_ffi_abi.int>
}

// PARSE-LABEL: func.func @function(
// PARSE: tvm_ffi_api.upcast
// PARSE: tvm_ffi_api.downcast
// FOLD-LABEL: func.func @function(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @function(%arg: !tvm_ffi_abi.function) -> !tvm_ffi_abi.function {
  %wide = tvm_ffi_api.upcast %arg : !tvm_ffi_abi.function -> !tvm_ffi_abi.function
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_abi.function -> !tvm_ffi_abi.function
  return %narrow : !tvm_ffi_abi.function
}

// PARSE-LABEL: func.func @int(
// PARSE: tvm_ffi_api.upcast
// PARSE: tvm_ffi_api.downcast
// FOLD-LABEL: func.func @int(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @int(%arg: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  %wide = tvm_ffi_api.upcast %arg : !tvm_ffi_abi.int -> !tvm_ffi_abi.int
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_abi.int -> !tvm_ffi_abi.int
  return %narrow : !tvm_ffi_abi.int
}

// PARSE-LABEL: func.func @union(
// PARSE: tvm_ffi_api.upcast
// PARSE: tvm_ffi_api.downcast
// FOLD-LABEL: func.func @union(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]:
// FOLD-NEXT: return %[[INPUT]]
func.func @union(%arg: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> {
  %wide = tvm_ffi_api.upcast %arg : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  return %narrow : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
}

// A block argument has no defining UpcastOp; narrowing must keep its assertion.
// PARSE-LABEL: func.func @narrow_argument(
// PARSE: tvm_ffi_api.downcast
// FOLD-LABEL: func.func @narrow_argument(
// FOLD-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi_api.any)
// FOLD-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_api.downcast %[[ARG]] : !tvm_ffi_api.any -> !tvm_ffi_abi.int
// FOLD-NEXT: return %[[VALUE]]
func.func @narrow_argument(%arg: !tvm_ffi_api.any) -> !tvm_ffi_abi.int {
  %value = tvm_ffi_api.downcast %arg : !tvm_ffi_api.any -> !tvm_ffi_abi.int
  return %value : !tvm_ffi_abi.int
}

// A successful widening proves the matching narrowing cannot trap.
// FOLD-LABEL: func.func @widen_narrow(
// FOLD-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// FOLD-NEXT: return %[[INPUT]]
func.func @widen_narrow(%input: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
  %wide = tvm_ffi_api.upcast %input : !tvm_ffi_abi.int -> !tvm_ffi_api.any
  %result = tvm_ffi_api.downcast %wide : !tvm_ffi_api.any -> !tvm_ffi_abi.int
  return %result : !tvm_ffi_abi.int
}

// A different target still requires its runtime assertion, even if unused.
// FOLD-LABEL: func.func @widen_narrow_mismatch(
// FOLD: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi_api.upcast
// FOLD: tvm_ffi_api.downcast %[[WIDE]] : !tvm_ffi_api.any -> !tvm_ffi_abi.float
// FOLD: return
func.func @widen_narrow_mismatch(%input: !tvm_ffi_abi.int) {
  %wide = tvm_ffi_api.upcast %input : !tvm_ffi_abi.int -> !tvm_ffi_api.any
  %result = tvm_ffi_api.downcast %wide : !tvm_ffi_api.any -> !tvm_ffi_abi.float
  return
}
