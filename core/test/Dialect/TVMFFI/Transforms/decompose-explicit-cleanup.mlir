// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.transfer'
// RUN: trident-core-opt %s --decompose-tvm-ffi --materialize-tvm-ffi-cleanup --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s --check-prefix=CLEAN --implicit-check-not='tvm_ffi.cleanup'

// Unmarked results deliberately transfer their references through return.
// Marked results retain exports before releasing local references.

// CHECK-LABEL: func.func @array_marked(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK: %[[CREATE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE]](%[[INPUT]]) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.c.array
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]](%[[ARRAY]], %[[INDEX:[a-zA-Z0-9_]+]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArraySize"
// CHECK-NEXT: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SIZE]](%[[ARRAY]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array) -> !tvm_ffi.int
// CLEAN-LABEL: func.func @array_marked(
// CLEAN: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]])
// CLEAN: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET:[a-zA-Z0-9_]+]](%[[ARRAY]], %[[INDEX:[a-zA-Z0-9_]+]])
// CLEAN: tvm_ffi.ObjectIncRef %[[ARRAY]]
// CLEAN-NEXT: tvm_ffi.ObjectIncRef %[[ELEMENT]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY]]
// CLEAN: return %[[ARRAY]], %[[ELEMENT]], %[[LENGTH:[a-zA-Z0-9_]+]]
func.func @array_marked(%input: !tvm_ffi.tensor) -> (!tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tensor, !tvm_ffi.int) {
  %array = tvm_ffi.array.create(%input) {tvm_ffi.cleanup} : (!tvm_ffi.tensor) -> !tvm_ffi.array<!tvm_ffi.tensor>
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %array[%index] {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.tensor
  %length = tvm_ffi.array.size %array {tvm_ffi.cleanup} : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.int
  return {tvm_ffi.retain} %array, %element, %length : !tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tensor, !tvm_ffi.int
}

// CHECK-LABEL: func.func @array_unmarked(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK: %[[CREATE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE]](%[[INPUT]]) : (!tvm_ffi.tensor) -> !tvm_ffi.c.array
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]](%[[ARRAY]], %[[INDEX:[a-zA-Z0-9_]+]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArraySize"
// CHECK-NEXT: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SIZE]](%[[ARRAY]]) : (!tvm_ffi.c.array) -> !tvm_ffi.int
// CLEAN-LABEL: func.func @array_unmarked(
// CLEAN: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]])
// CLEAN: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET:[a-zA-Z0-9_]+]](%[[ARRAY]], %[[INDEX:[a-zA-Z0-9_]+]])
// CLEAN-NOT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CLEAN-NOT: tvm_ffi.ObjectDecRef %[[ARRAY]]
// CLEAN: return %[[ARRAY]], %[[ELEMENT]], %[[LENGTH:[a-zA-Z0-9_]+]]
func.func @array_unmarked(%input: !tvm_ffi.tensor) -> (!tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tensor, !tvm_ffi.int) {
  %array = tvm_ffi.array.create(%input) : (!tvm_ffi.tensor) -> !tvm_ffi.array<!tvm_ffi.tensor>
  %index = arith.constant 0 : i64
  %element = tvm_ffi.array.get %array[%index] : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.tensor
  %length = tvm_ffi.array.size %array : !tvm_ffi.array<!tvm_ffi.tensor> -> !tvm_ffi.int
  return %array, %element, %length : !tvm_ffi.array<!tvm_ffi.tensor>, !tvm_ffi.tensor, !tvm_ffi.int
}

// CHECK-LABEL: func.func @call_marked(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.object, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[INPUT]]) {test.call, tvm_ffi.cleanup} : (!tvm_ffi.c.array) -> !tvm_ffi.c.array
// CLEAN-LABEL: func.func @call_marked(
// CLEAN: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]]) {test.call}
// CLEAN-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]
func.func @call_marked(%handle: !tvm_ffi.object, %input: !tvm_ffi.tuple<>) -> !tvm_ffi.tuple<> {
  %result = tvm_ffi.FunctionCall %handle(%input) {test.call, tvm_ffi.cleanup} : (!tvm_ffi.tuple<>) -> !tvm_ffi.tuple<>
  return {tvm_ffi.retain} %result : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @call_unmarked(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.object, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[INPUT]]) {test.call} : (!tvm_ffi.c.array) -> !tvm_ffi.c.array
// CLEAN-LABEL: func.func @call_unmarked(
// CLEAN: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]]) {test.call}
// CLEAN-NEXT: return %[[RESULT]]
func.func @call_unmarked(%handle: !tvm_ffi.object, %input: !tvm_ffi.tuple<>) -> !tvm_ffi.tuple<> {
  %result = tvm_ffi.FunctionCall %handle(%input) {test.call} : (!tvm_ffi.tuple<>) -> !tvm_ffi.tuple<>
  return %result : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @empty_marked(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]() {tvm_ffi.cleanup} : () -> !tvm_ffi.c.array
// CLEAN-LABEL: func.func @empty_marked(
// CLEAN: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE:[a-zA-Z0-9_]+]]()
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CLEAN-NEXT: tvm_ffi.ObjectIncRef %[[EMPTY]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[EMPTY]]
// CLEAN-NEXT: return %[[EMPTY]]
func.func @empty_marked() -> !tvm_ffi.tuple<> {
  %empty = tvm_ffi.tuple.construct() {tvm_ffi.cleanup} : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %empty {tvm_ffi.cleanup} : (!tvm_ffi.tuple<>) -> ()
  return {tvm_ffi.retain} %empty : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @empty_unmarked(
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]() : () -> !tvm_ffi.c.array
// CLEAN-LABEL: func.func @empty_unmarked(
// CLEAN: %[[EMPTY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE:[a-zA-Z0-9_]+]]()
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CLEAN-NEXT: return %[[EMPTY]]
func.func @empty_unmarked() -> !tvm_ffi.tuple<> {
  %empty = tvm_ffi.tuple.construct() : () -> !tvm_ffi.tuple<>
  tvm_ffi.tuple.unconstruct %empty : (!tvm_ffi.tuple<>) -> ()
  return %empty : !tvm_ffi.tuple<>
}

// CHECK-LABEL: func.func @tuple_marked(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[INT:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK: %[[CREATE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE]](%[[INPUT]], %[[INT]]) {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[GET0:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET0]](%[[TUPLE]], %[[INDEX0:[a-zA-Z0-9_]+]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK: %[[GET1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[INTEGER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET1]](%[[TUPLE]], %[[INDEX1:[a-zA-Z0-9_]+]]) {tvm_ffi.cleanup} : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CLEAN-LABEL: func.func @tuple_marked(
// CLEAN: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]], %[[INT:[a-zA-Z0-9_]+]])
// CLEAN: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET0:[a-zA-Z0-9_]+]](%[[TUPLE]], %[[INDEX0:[a-zA-Z0-9_]+]])
// CLEAN: %[[INTEGER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET1:[a-zA-Z0-9_]+]](%[[TUPLE]], %[[INDEX1:[a-zA-Z0-9_]+]])
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[GET1]]
// CLEAN-NEXT: tvm_ffi.ObjectIncRef %[[TUPLE]]
// CLEAN-NEXT: tvm_ffi.ObjectIncRef %[[ELEMENT]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[ELEMENT]]
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE]]
// CLEAN-NEXT: return %[[TUPLE]], %[[ELEMENT]], %[[INTEGER]]
func.func @tuple_marked(%input: !tvm_ffi.tensor, %integer: !tvm_ffi.int) -> (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>, !tvm_ffi.tensor, !tvm_ffi.int) {
  %tuple = tvm_ffi.tuple.construct(%input, %integer) {tvm_ffi.cleanup} : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>
  %element, %value = tvm_ffi.tuple.unconstruct %tuple {tvm_ffi.cleanup} : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>) -> (!tvm_ffi.tensor, !tvm_ffi.int)
  return {tvm_ffi.retain} %tuple, %element, %value : !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>, !tvm_ffi.tensor, !tvm_ffi.int
}

// CHECK-LABEL: func.func @tuple_unmarked(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[INT:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK: %[[CREATE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE]](%[[INPUT]], %[[INT]]) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[GET0:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET0]](%[[TUPLE]], %[[INDEX0:[a-zA-Z0-9_]+]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK: %[[GET1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[INTEGER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET1]](%[[TUPLE]], %[[INDEX1:[a-zA-Z0-9_]+]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CLEAN-LABEL: func.func @tuple_unmarked(
// CLEAN: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE:[a-zA-Z0-9_]+]](%[[INPUT:[a-zA-Z0-9_]+]], %[[INT:[a-zA-Z0-9_]+]])
// CLEAN: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET0:[a-zA-Z0-9_]+]](%[[TUPLE]], %[[INDEX0:[a-zA-Z0-9_]+]])
// CLEAN: %[[INTEGER:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET1:[a-zA-Z0-9_]+]](%[[TUPLE]], %[[INDEX1:[a-zA-Z0-9_]+]])
// CLEAN-NEXT: tvm_ffi.ObjectDecRef %[[GET1]]
// CLEAN-NEXT: return %[[TUPLE]], %[[ELEMENT]], %[[INTEGER]]
func.func @tuple_unmarked(%input: !tvm_ffi.tensor, %integer: !tvm_ffi.int) -> (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>, !tvm_ffi.tensor, !tvm_ffi.int) {
  %tuple = tvm_ffi.tuple.construct(%input, %integer) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>
  %element, %value = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>) -> (!tvm_ffi.tensor, !tvm_ffi.int)
  return %tuple, %element, %value : !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.int>, !tvm_ffi.tensor, !tvm_ffi.int
}
