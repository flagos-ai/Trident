// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield' --implicit-check-not='unrealized_conversion_cast'
// RUN: trident-core-opt %s --decompose-tvm-ffi --decompose-tvm-ffi --verify-each | FileCheck %s

// Container calls must stay inside the successful continuation, including calls
// whose operands are lexical captures rather than bind payloads.
// CHECK-LABEL: func.func @container_continuation(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.array>, %[[CAPTURE:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK-NOT: tvm_ffi.FunctionGetGlobal
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[OK]] -> (!tvm_ffi.monad.result<!tvm_ffi.c.array>) {
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.c.array
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]](%[[ARRAY]], %[[CAPTURE]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[TUPLE]] : (!tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: scf.yield %[[PURE]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.Function
// CHECK: %[[FAIL:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: scf.yield %[[FAIL]]
// CHECK: return %[[BRANCH]]
func.func @container_continuation(%input: !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>, %capture: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>> {
  ^bb0(%array: !tvm_ffi.array<!tvm_ffi.int>):
    %tuple = tvm_ffi.tuple.construct(%array, %capture) : (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>
    %pure = tvm_ffi.monad.pure %tuple : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
  }
  return %result : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>
}

// Nested continuations combine remapped block arguments, lexical captures and
// structural SCF conversion. No semantic container cast may escape conversion.
// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.array>, %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.int>, %[[INDEX:[a-zA-Z0-9_]+]]: i64, %[[PREDICATE:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if %[[OK]]
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.c.array
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ZERO]]
// CHECK: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FIRST_HANDLE]](%[[TUPLE]], %[[FIRST]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.c.array
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = arith.constant 1 : i64
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[ONE]]
// CHECK: %[[SECOND_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[CAPTURE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SECOND_HANDLE]](%[[TUPLE]], %[[SECOND]]) : (!tvm_ffi.c.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if %[[PREDICATE]] -> (!tvm_ffi.c.array)
// CHECK: scf.yield %[[ARRAY]]
// CHECK: } else {
// CHECK: %[[FALLBACK_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[FALLBACK:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FALLBACK_HANDLE]](%[[CAPTURE]])
// CHECK: scf.yield %[[FALLBACK]]
// CHECK: %[[SCALAR_OK:[a-zA-Z0-9_]+]], %[[SCALAR_ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[SCALAR]]
// CHECK: %[[INNER:[a-zA-Z0-9_]+]] = scf.if %[[SCALAR_OK]]
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[SCALAR_ANY]] : !tvm_ffi.any -> !tvm_ffi.int
// CHECK: %[[BOXED_INDEX:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INDEX]]
// CHECK: %[[GET:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[GET]](%[[CHOSEN]], %[[BOXED_INDEX]])
// CHECK: %[[CREATE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[CREATE]](%[[ITEM]], %[[VALUE]], %[[CAPTURE]])
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[RESULT]]
// CHECK: scf.yield %[[PURE]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.Function
// CHECK: %[[INNER_FAIL:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[INNER_FAIL]]
// CHECK: scf.yield %[[INNER]]
// CHECK: } else {
// CHECK-NOT: tvm_ffi.Function
// CHECK: %[[OUTER_FAIL:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail
// CHECK: scf.yield %[[OUTER_FAIL]]
// CHECK: return %[[OUTER]]
func.func @nested_containers(%input: !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>, %scalar: !tvm_ffi.monad.result<!tvm_ffi.int>, %index: i64, %predicate: i1) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  %outer = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  ^bb0(%tuple: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>):
    %array, %capture = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int>) -> (!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.int)
    %chosen = scf.if %predicate -> (!tvm_ffi.array<!tvm_ffi.int>) {
      scf.yield %array : !tvm_ffi.array<!tvm_ffi.int>
    } else {
      %fallback = tvm_ffi.array.create(%capture) : (!tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
      scf.yield %fallback : !tvm_ffi.array<!tvm_ffi.int>
    }
    %inner = tvm_ffi.monad.bind %scalar : (!tvm_ffi.monad.result<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
    ^bb1(%value: !tvm_ffi.int):
      %item = tvm_ffi.array.get %chosen[%index] : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.int
      %result = tvm_ffi.array.create(%item, %value, %capture) : (!tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array<!tvm_ffi.int>
      %pure = tvm_ffi.monad.pure %result : (!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
      tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
    }
    tvm_ffi.monad.yield %inner : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
  }
  return %outer : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
}
