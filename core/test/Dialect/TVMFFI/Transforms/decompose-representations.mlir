// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.any' --implicit-check-not='!tvm_ffi.union<' --implicit-check-not='!tvm_ffi.str'

// CHECK-LABEL: func.func @narrow_union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
func.func @narrow_union(%value: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none>) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %narrow = tvm_ffi.downcast %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float, !tvm_ffi.none> -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %narrow : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// CHECK-LABEL: func.func @unused_string_downcast(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// CHECK-NEXT: return
func.func @unused_string_downcast(%value: !tvm_ffi.any) {
  %string = tvm_ffi.downcast %value : !tvm_ffi.any -> !tvm_ffi.str
  return
}

// Result wrappers and unpacked dynamic payloads convert recursively.
// CHECK-LABEL: func.func @result_payload(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.any>) -> !tvm_ffi.c.any
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]] : (!tvm_ffi.monad.result<!tvm_ffi.c.any>) -> (i1, !tvm_ffi.c.any)
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[PAYLOAD]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[PAYLOAD]] : !tvm_ffi.c.any
func.func @result_payload(%input: !tvm_ffi.monad.result<!tvm_ffi.any>) -> !tvm_ffi.any {
  %ok, %payload = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  return %payload : !tvm_ffi.any
}

// Finite boundary constraints are currently erased to unconstrained CAny.
// CHECK-LABEL: tvm_ffi.func @boundary(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[VALUE]] : !tvm_ffi.c.any -> !tvm_ffi.c.any
// CHECK: tvm_ffi.ObjectIncRef %[[PAYLOAD]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.return %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD]]
tvm_ffi.func @boundary(%value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.str>) -> !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.str> {
  %payload = tvm_ffi.upcast %value : !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.str> -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
}

// -----

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @identity(%[[INPUT]], %[[SECOND]]) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}]} : (!tvm_ffi.c.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[OK]], %[[VALUE]]
// CHECK-LABEL: func.func @invoke_empty(
// CHECK-NEXT: %[[EMPTY_OK:[a-zA-Z0-9_]+]], %[[EMPTY_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK: return %[[EMPTY_OK]], %[[EMPTY_VALUE]]
// CHECK-LABEL: tvm_ffi.func @identity(
// CHECK-SAME: !tvm_ffi.c.tensor,
// CHECK-SAME: !tvm_ffi.int
// CHECK-LABEL: tvm_ffi.func @empty(

func.func @invoke(%input: !tvm_ffi.c.tensor, %second: !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @identity(%input, %second) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}]} : (!tvm_ffi.c.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
func.func @invoke_empty() -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
tvm_ffi.func @identity(%input: !tvm_ffi.c.tensor, %second: !tvm_ffi.int) -> !tvm_ffi.c.tensor {
  %any = tvm_ffi.upcast %input : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}
tvm_ffi.func @empty() -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
  %ok = tvm_ffi.constant.bool true
  tvm_ffi.return %ok, %any : !tvm_ffi.bool, !tvm_ffi.any
}

// CHECK-LABEL: func.func @invoke_container(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK: %[[CONTAINER_OK:[a-zA-Z0-9_]+]], %[[CONTAINER_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @container(%[[ARG]]) {test.attribute} : (!tvm_ffi.c.array) -> (!tvm_ffi.bool, !tvm_ffi.c.any)
// CHECK: return %[[CONTAINER_OK]], %[[CONTAINER_VALUE]]
func.func @invoke_container(%arg: !tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @container(%arg) {test.attribute} : (!tvm_ffi.array<!tvm_ffi.int>) -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}
// CHECK-LABEL: tvm_ffi.func @container(!tvm_ffi.c.array) -> !tvm_ffi.c.array
tvm_ffi.func @container(!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.tuple<!tvm_ffi.int>

// -----

// CHECK-LABEL: func.func @objects(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor,
// CHECK-SAME: %[[FUNCTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function,
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi.c.exception)
// CHECK: return %[[TENSOR]], %[[FUNCTION]], %[[EXCEPTION]] : !tvm_ffi.c.tensor, !tvm_ffi.c.function, !tvm_ffi.c.exception
func.func @objects(%tensor: !tvm_ffi.tensor, %function: !tvm_ffi.function, %exception: !tvm_ffi.exception) -> (!tvm_ffi.tensor, !tvm_ffi.function, !tvm_ffi.exception) {
  return %tensor, %function, %exception : !tvm_ffi.tensor, !tvm_ffi.function, !tvm_ffi.exception
}

// CHECK-LABEL: func.func @lookup(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.c.function
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function() : () -> !tvm_ffi.c.array
// CHECK: return %[[ARRAY]] : !tvm_ffi.c.array
func.func @lookup() -> !tvm_ffi.array<!tvm_ffi.any> {
  %handle = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
  %array = tvm_ffi.FunctionCall %handle : !tvm_ffi.function() : () -> !tvm_ffi.array<!tvm_ffi.any>
  return %array : !tvm_ffi.array<!tvm_ffi.any>
}

// CHECK-LABEL: func.func @strings(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// CHECK: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "hello"
// CHECK: %[[STRING:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW]] : !tvm_ffi.raw_str -> !tvm_ffi.c.any
// CHECK: return %[[STRING]], %[[VALUE]]
func.func @strings(%value: !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>) -> (!tvm_ffi.str, !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>) {
  %text = tvm_ffi.constant.str "hello"
  return %text, %value : !tvm_ffi.str, !tvm_ffi.union<!tvm_ffi.str, !tvm_ffi.raw_str>
}

// CHECK-LABEL: func.func @result_and_attributes(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.tensor>)
// CHECK-SAME: attributes {test.types = [!tvm_ffi.c.exception, (!tvm_ffi.c.tensor) -> !tvm_ffi.c.function]}
// CHECK: return %[[RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
func.func @result_and_attributes(%value: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.tensor> attributes {test.types = [!tvm_ffi.exception, (!tvm_ffi.tensor) -> !tvm_ffi.function]} {
  return %value : !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// -----

// Cast chains borrow their source; exports retain the semantic source once per
// occurrence without adding local cleanup responsibility for each cast.
// CHECK-LABEL: func.func @cast_exports(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[OWNED:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[OWNED]]
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK: %[[EXPORT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[NARROW]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[EXPORT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[EXPORT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OWNED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[EXPORT]], %[[EXPORT]]
func.func @cast_exports(%input: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.any) {
  %owned = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %wide = tvm_ffi.upcast %owned : !tvm_ffi.tensor -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.tensor
  %export = tvm_ffi.upcast %narrow : !tvm_ffi.tensor -> !tvm_ffi.any
  return %export, %export : !tvm_ffi.any, !tvm_ffi.any
}

// Every Opaque export is retained, including casts from Base references and POD.
// The casts borrow their inputs and register no local cleanup.
// CHECK-LABEL: func.func @opaque_exports(
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[POD:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[BASE]]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[POD]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OBJECT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SCALAR]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[OBJECT]], %[[SCALAR]]
func.func @opaque_exports(%base: !tvm_ffi.c.tensor, %pod: !tvm_ffi.int) -> (!tvm_ffi.any, !tvm_ffi.any) {
  %object = tvm_ffi.upcast %base : !tvm_ffi.c.tensor -> !tvm_ffi.any
  %scalar = tvm_ffi.upcast %pod : !tvm_ffi.int -> !tvm_ffi.any
  return %object, %scalar : !tvm_ffi.any, !tvm_ffi.any
}

// Array and tuple extraction must use the same expanded string representation.
// CHECK-LABEL: func.func @string_elements(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]] : i64 -> !tvm_ffi.int
// CHECK: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ARRAY_HANDLE]] : !tvm_ffi.c.function(%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY_HANDLE]] : !tvm_ffi.c.function
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO]] : i64 -> !tvm_ffi.int
// CHECK: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[TUPLE_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[OFFSET]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]] : !tvm_ffi.c.any
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.any
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
func.func @string_elements(%array: !tvm_ffi.array<!tvm_ffi.str>, %tuple: !tvm_ffi.tuple<!tvm_ffi.str>, %index: i64) -> (!tvm_ffi.str, !tvm_ffi.str) {
  %first = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.str> -> !tvm_ffi.str
  %second = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.str>) -> !tvm_ffi.str
  return %first, %second : !tvm_ffi.str, !tvm_ffi.str
}

// Even unions whose members share a representation lower to CAny.
// CHECK-LABEL: func.func @union_collapse(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any) -> !tvm_ffi.c.any
// CHECK: return %[[VALUE]] : !tvm_ffi.c.any
func.func @union_collapse(%value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor>) -> !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor> {
  return %value : !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.c.tensor>
}

// -----

// CHECK-LABEL: func.func @test_exception_tag(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = tvm_ffi.isa %[[VALUE]] is !tvm_ffi.c.exception
// CHECK-NEXT: return %[[MATCH]]
func.func @test_exception_tag(%value: !tvm_ffi.any) -> i1 {
  %match = tvm_ffi.isa %value is !tvm_ffi.exception : !tvm_ffi.any
  return %match : i1
}
