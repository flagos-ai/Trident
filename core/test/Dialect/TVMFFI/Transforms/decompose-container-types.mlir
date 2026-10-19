// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.array<' --implicit-check-not='!tvm_ffi.tuple<' --implicit-check-not='tvm_ffi.monad.bind' --implicit-check-not='tvm_ffi.monad.yield'

// CHECK-LABEL: tvm_ffi.func @bind_container(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.monad.result<!tvm_ffi.c.array>) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: %[[OK:[a-zA-Z0-9_]+]], %[[ANY:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT]]
// CHECK: %[[BRANCH:[a-zA-Z0-9_]+]] = scf.if %[[OK]] -> (!tvm_ffi.monad.result<!tvm_ffi.c.array>)
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[ANY]] : !tvm_ffi.any -> !tvm_ffi.c.array
// CHECK: %[[PURE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[PAYLOAD]] : (!tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: scf.yield %[[PURE]]
// CHECK: %[[FAIL:[a-zA-Z0-9_]+]] = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: scf.yield %[[FAIL]]
// CHECK: tvm_ffi.return %[[BRANCH]]
tvm_ffi.func @bind_container(%input: !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  %result = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>> {
  ^bb0(%payload: !tvm_ffi.array<!tvm_ffi.int>):
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
  }
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.array<!tvm_ffi.int>>
}

// CHECK-LABEL: func.func @casts(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[ARRAY]] : !tvm_ffi.c.array -> !tvm_ffi.any {test.keep = "wide"}
// CHECK: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]] : !tvm_ffi.any -> !tvm_ffi.c.array {test.types = [!tvm_ffi.c.array]}
// CHECK: return %[[NARROW]]
func.func @casts(%array: !tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.array<!tvm_ffi.int> {
  %wide = tvm_ffi.upcast %array : !tvm_ffi.array<!tvm_ffi.int> -> !tvm_ffi.any {test.keep = "wide"}
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.array<!tvm_ffi.int> {test.types = [!tvm_ffi.tuple<!tvm_ffi.int>]}
  return %narrow : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @conditional_branch(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[PREDICATE:[a-zA-Z0-9_]+]]: i1)
// CHECK: cf.cond_br %[[PREDICATE]] weights([3, 1]), ^[[TAKEN_BLOCK:bb[0-9]+]](%[[VALUE]] : !tvm_ffi.c.array), ^[[OTHER_BLOCK:bb[0-9]+]]
// CHECK: ^[[TAKEN_BLOCK]](%[[TAKEN:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array):
// CHECK: return %[[TAKEN]]
// CHECK: ^[[OTHER_BLOCK]]:
// CHECK: return %[[VALUE]]
func.func @conditional_branch(%value: !tvm_ffi.array<!tvm_ffi.int>, %predicate: i1) -> !tvm_ffi.array<!tvm_ffi.int> {
  cf.cond_br %predicate, ^taken(%value : !tvm_ffi.array<!tvm_ffi.int>), ^other {branch_weights = array<i32: 3, 1>}
^taken(%taken: !tvm_ffi.array<!tvm_ffi.int>):
  return %taken : !tvm_ffi.array<!tvm_ffi.int>
^other:
  return %value : !tvm_ffi.array<!tvm_ffi.int>
}

// CHECK-LABEL: func.func @nested_containers(
// CHECK-SAME: %[[NESTED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.monad.pure %[[NESTED]] {test.types = [!tvm_ffi.c.array, !tvm_ffi.monad.result<!tvm_ffi.c.array>]} : (!tvm_ffi.c.array) -> !tvm_ffi.monad.result<!tvm_ffi.c.array>
// CHECK: return %[[RESULT]] : !tvm_ffi.monad.result<!tvm_ffi.c.array>
func.func @nested_containers(%nested: !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>> {
  %result = tvm_ffi.monad.pure %nested {test.types = [!tvm_ffi.array<!tvm_ffi.int>, !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.int>>]} : (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>) -> !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>>
  return %result : !tvm_ffi.monad.result<!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.tuple<!tvm_ffi.int>>>>
}

// CHECK-LABEL: func.func @opaque_passthrough(
// CHECK-SAME: %[[ANY:[a-zA-Z0-9_]+]]: !tvm_ffi.any,
// CHECK-SAME: %[[EXCEPTION:[a-zA-Z0-9_]+]]: !tvm_ffi.exception,
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !tvm_ffi.object,
// CHECK-SAME: %[[UNION:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>)
// CHECK: return %[[ANY]], %[[EXCEPTION]], %[[OBJECT]], %[[UNION]] : !tvm_ffi.any, !tvm_ffi.exception, !tvm_ffi.object, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
func.func @opaque_passthrough(%any: !tvm_ffi.any, %exception: !tvm_ffi.exception, %object: !tvm_ffi.object, %union: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) -> (!tvm_ffi.any, !tvm_ffi.exception, !tvm_ffi.object, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>) {
  return %any, %exception, %object, %union : !tvm_ffi.any, !tvm_ffi.exception, !tvm_ffi.object, !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// CHECK-LABEL: func.func @while_container(
// CHECK-SAME: %[[INITIAL:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array,
// CHECK-SAME: %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[LOOP:[a-zA-Z0-9_]+]] = scf.while (%[[CURRENT:[a-zA-Z0-9_]+]] = %[[INITIAL]]) : (!tvm_ffi.c.array) -> !tvm_ffi.c.array
// CHECK: scf.condition(%[[READY]]) %[[CURRENT]] : !tvm_ffi.c.array
// CHECK: ^[[BODY:bb[0-9]+]](%[[NEXT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array):
// CHECK: scf.yield %[[NEXT]] : !tvm_ffi.c.array
// CHECK: return %[[LOOP]]
func.func @while_container(%initial: !tvm_ffi.array<!tvm_ffi.int>, %ready: i1) -> !tvm_ffi.array<!tvm_ffi.int> {
  %result = scf.while (%current = %initial) : (!tvm_ffi.array<!tvm_ffi.int>) -> !tvm_ffi.array<!tvm_ffi.int> {
    scf.condition(%ready) %current : !tvm_ffi.array<!tvm_ffi.int>
  } do {
  ^bb0(%next: !tvm_ffi.array<!tvm_ffi.int>):
    scf.yield %next : !tvm_ffi.array<!tvm_ffi.int>
  }
  return %result : !tvm_ffi.array<!tvm_ffi.int>
}
