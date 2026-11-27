// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --split-input-file --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi_api.any' --implicit-check-not='!tvm_ffi_api.union<' --implicit-check-not='!tvm_ffi_api.str'

// CHECK-LABEL: func.func @narrow_union(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi_abi.any
func.func @narrow_union(%value: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none>) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> {
  %narrow = tvm_ffi_api.downcast %value : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float, !tvm_ffi_abi.none> -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  return %narrow : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
}

// CHECK-LABEL: func.func @unused_string_downcast(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any)
// CHECK-NEXT: return
func.func @unused_string_downcast(%value: !tvm_ffi_api.any) {
  %string = tvm_ffi_api.downcast %value : !tvm_ffi_api.any -> !tvm_ffi_api.str
  return
}

// Result wrappers and unpacked dynamic payloads convert recursively.
// CHECK-LABEL: func.func @result_payload(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.any>) -> !tvm_ffi_abi.any
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_monad.unpack %[[INPUT]] : (!tvm_ffi_monad.result<!tvm_ffi_abi.any>) -> (i1, !tvm_ffi_abi.any)
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[PAYLOAD]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[PAYLOAD]] : !tvm_ffi_abi.any
func.func @result_payload(%input: !tvm_ffi_monad.result<!tvm_ffi_api.any>) -> !tvm_ffi_api.any {
  %ok, %payload = tvm_ffi_monad.unpack %input : (!tvm_ffi_monad.result<!tvm_ffi_api.any>) -> (i1, !tvm_ffi_api.any)
  return %payload : !tvm_ffi_api.any
}

// Finite boundary constraints are currently erased to unconstrained CAny.
// CHECK-LABEL: tvm_ffi_abi.func @boundary(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[VALUE]] : !tvm_ffi_abi.any -> !tvm_ffi_abi.any
// CHECK: tvm_ffi_abi.object.inc_ref %[[PAYLOAD]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.return %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD]]
tvm_ffi_api.func @boundary(%value: !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_api.str>) -> !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_api.str> {
  %payload = tvm_ffi_api.upcast %value : !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_api.str> -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// -----

// CHECK-LABEL: func.func @invoke(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.call @identity(%[[INPUT]], %[[SECOND]]) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}]} : (!tvm_ffi_abi.tensor, !tvm_ffi_abi.int) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[OK]], %[[VALUE]]
// CHECK-LABEL: func.func @invoke_empty(
// CHECK-NEXT: %[[EMPTY_OK:[a-zA-Z0-9_]+]], %[[EMPTY_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.call @empty() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK: return %[[EMPTY_OK]], %[[EMPTY_VALUE]]
// CHECK-LABEL: tvm_ffi_abi.func @identity(
// CHECK-SAME: !tvm_ffi_abi.tensor,
// CHECK-SAME: !tvm_ffi_abi.int
// CHECK-LABEL: tvm_ffi_abi.func @empty(

func.func @invoke(%input: !tvm_ffi_abi.tensor, %second: !tvm_ffi_abi.int) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_api.call @identity(%input, %second) {arg_attrs = [{test.argument}, {}], res_attrs = [{test.success}, {test.payload}]} : (!tvm_ffi_abi.tensor, !tvm_ffi_abi.int) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
func.func @invoke_empty() -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_api.call @empty() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
tvm_ffi_api.func @identity(%input: !tvm_ffi_abi.tensor, %second: !tvm_ffi_abi.int) -> !tvm_ffi_abi.tensor {
  %any = tvm_ffi_api.upcast %input : !tvm_ffi_abi.tensor -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
tvm_ffi_api.func @empty() -> !tvm_ffi_abi.none {
  %none = tvm_ffi_abi.constant.none
  %any = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  tvm_ffi_api.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// CHECK-LABEL: func.func @invoke_container(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK: %[[CONTAINER_OK:[a-zA-Z0-9_]+]], %[[CONTAINER_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.call @container(%[[ARG]]) {test.attribute} : (!tvm_ffi_abi.array) -> (!tvm_ffi_abi.bool, !tvm_ffi_abi.any)
// CHECK: return %[[CONTAINER_OK]], %[[CONTAINER_VALUE]]
func.func @invoke_container(%arg: !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_api.call @container(%arg) {test.attribute} : (!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}
// CHECK-LABEL: tvm_ffi_abi.func @container(!tvm_ffi_abi.array) -> !tvm_ffi_abi.array
tvm_ffi_api.func @container(!tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.int>

// -----

// CHECK-LABEL: func.func @objects(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor,
// CHECK-SAME: %[[FUNCTION:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function,
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.exception)
// CHECK: return %[[TENSOR]], %[[FUNCTION]], %[[EXCEPTION]] : !tvm_ffi_abi.tensor, !tvm_ffi_abi.function, !tvm_ffi_abi.exception
func.func @objects(%tensor: !tvm_ffi_api.tensor, %function: !tvm_ffi_api.function, %exception: !tvm_ffi_api.exception) -> (!tvm_ffi_api.tensor, !tvm_ffi_api.function, !tvm_ffi_api.exception) {
  return %tensor, %function, %exception : !tvm_ffi_api.tensor, !tvm_ffi_api.function, !tvm_ffi_api.exception
}

// CHECK-LABEL: func.func @lookup(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.Array" : !tvm_ffi_abi.function
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[HANDLE]] : !tvm_ffi_abi.function() : () -> !tvm_ffi_abi.array
// CHECK: return %[[ARRAY]] : !tvm_ffi_abi.array
func.func @lookup() -> !tvm_ffi_api.array<!tvm_ffi_api.any> {
  %handle = tvm_ffi_api.function.get_global "ffi.Array" : !tvm_ffi_api.function
  %array = tvm_ffi_api.function.call %handle : !tvm_ffi_api.function() : () -> !tvm_ffi_api.array<!tvm_ffi_api.any>
  return %array : !tvm_ffi_api.array<!tvm_ffi_api.any>
}

// CHECK-LABEL: func.func @strings(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any)
// CHECK: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.raw_str "hello"
// CHECK: %[[STRING:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[RAW]] : !tvm_ffi_abi.raw_str -> !tvm_ffi_abi.any
// CHECK: return %[[STRING]], %[[VALUE]]
func.func @strings(%value: !tvm_ffi_api.union<!tvm_ffi_api.str, !tvm_ffi_abi.raw_str>) -> (!tvm_ffi_api.str, !tvm_ffi_api.union<!tvm_ffi_api.str, !tvm_ffi_abi.raw_str>) {
  %text = tvm_ffi_api.constant.str "hello"
  return %text, %value : !tvm_ffi_api.str, !tvm_ffi_api.union<!tvm_ffi_api.str, !tvm_ffi_abi.raw_str>
}

// CHECK-LABEL: func.func @result_and_attributes(
// CHECK-SAME: %[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>)
// CHECK-SAME: attributes {test.types = [!tvm_ffi_abi.exception, (!tvm_ffi_abi.tensor) -> !tvm_ffi_abi.function]}
// CHECK: return %[[RESULT]] : !tvm_ffi_monad.result<!tvm_ffi_abi.tensor>
func.func @result_and_attributes(%value: !tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor> attributes {test.types = [!tvm_ffi_api.exception, (!tvm_ffi_api.tensor) -> !tvm_ffi_api.function]} {
  return %value : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
}

// -----

// Cast chains borrow their source; exports retain the semantic source once per
// occurrence without adding local cleanup responsibility for each cast.
// CHECK-LABEL: func.func @cast_exports(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[OWNED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[OWNED]]
// CHECK: %[[EXPORT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[OWNED]]
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[EXPORT]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[EXPORT]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[OWNED]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: return %[[EXPORT]], %[[EXPORT]]
func.func @cast_exports(%input: !tvm_ffi_api.tensor) -> (!tvm_ffi_api.any, !tvm_ffi_api.any) {
  %owned = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
  %wide = tvm_ffi_api.upcast %owned : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_api.any -> !tvm_ffi_api.tensor
  %export = tvm_ffi_api.upcast %narrow : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
  return %export, %export : !tvm_ffi_api.any, !tvm_ffi_api.any
}

// Every Opaque export is retained, including casts from Base references and POD.
// The casts borrow their inputs and register no local cleanup.
// CHECK-LABEL: func.func @opaque_exports(
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[POD:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.int)
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[BASE]]
// CHECK-NEXT: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[POD]]
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[OBJECT]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[SCALAR]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[OBJECT]], %[[SCALAR]]
func.func @opaque_exports(%base: !tvm_ffi_abi.tensor, %pod: !tvm_ffi_abi.int) -> (!tvm_ffi_api.any, !tvm_ffi_api.any) {
  %object = tvm_ffi_api.upcast %base : !tvm_ffi_abi.tensor -> !tvm_ffi_api.any
  %scalar = tvm_ffi_api.upcast %pod : !tvm_ffi_abi.int -> !tvm_ffi_api.any
  return %object, %scalar : !tvm_ffi_api.any, !tvm_ffi_api.any
}

// Array and tuple extraction must use the same expanded string representation.
// CHECK-LABEL: func.func @string_elements(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[BOXED:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[INDEX]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[ARRAY_HANDLE]] : !tvm_ffi_abi.function(%[[ARRAY]], %[[BOXED]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[ARRAY_HANDLE]] : !tvm_ffi_abi.function
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi_abi.to %[[ZERO]] : i64 -> !tvm_ffi_abi.int
// CHECK: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.get_global "ffi.ArrayGetItem"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.function.call %[[TUPLE_HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[OFFSET]]) : (!tvm_ffi_abi.array, !tvm_ffi_abi.int) -> !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.dec_ref %[[TUPLE_HANDLE]] : !tvm_ffi_abi.function
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[FIRST]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.object.inc_ref %[[SECOND]] : !tvm_ffi_abi.any
// CHECK-DAG: tvm_ffi_abi.object.dec_ref %[[SECOND]] : !tvm_ffi_abi.any
// CHECK-DAG: tvm_ffi_abi.object.dec_ref %[[FIRST]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
func.func @string_elements(%array: !tvm_ffi_api.array<!tvm_ffi_api.str>, %tuple: !tvm_ffi_api.tuple<!tvm_ffi_api.str>, %index: i64) -> (!tvm_ffi_api.str, !tvm_ffi_api.str) {
  %first = tvm_ffi_api.array.get %array[%index] : !tvm_ffi_api.array<!tvm_ffi_api.str> -> !tvm_ffi_api.str
  %second = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.str>) -> !tvm_ffi_api.str
  return %first, %second : !tvm_ffi_api.str, !tvm_ffi_api.str
}

// Even unions whose members share a representation lower to CAny.
// CHECK-LABEL: func.func @union_collapse(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any) -> !tvm_ffi_abi.any
// CHECK: return %[[VALUE]] : !tvm_ffi_abi.any
func.func @union_collapse(%value: !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor>) -> !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor> {
  return %value : !tvm_ffi_api.union<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor>
}

// -----

// CHECK-LABEL: func.func @test_exception_tag(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any
// CHECK: %[[MATCH:[a-zA-Z0-9_]+]] = tvm_ffi_abi.isa %[[VALUE]] is !tvm_ffi_abi.exception
// CHECK-NEXT: return %[[MATCH]]
func.func @test_exception_tag(%value: !tvm_ffi_api.any) -> i1 {
  %match = tvm_ffi_api.isa %value is !tvm_ffi_api.exception : !tvm_ffi_api.any
  return %match : i1
}
