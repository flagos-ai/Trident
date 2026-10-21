// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --verify-each | FileCheck %s --check-prefix=DEFER
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'

tvm_ffi.func @empty() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}

// DEFER-LABEL: func.func @aliases(
// DEFER-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER: return

// CHECK-LABEL: func.func @aliases(
// CHECK-SAME: %[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[BORROWED]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[BORROWED]]
// CHECK: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ALIAS]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ALIAS]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BORROWED]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[ALIAS]], %[[ALIAS]], %[[BORROWED]]
func.func @aliases(%input: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.any, !tvm_ffi.tensor) {
  %first = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %second = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  %alias = tvm_ffi.upcast %first : !tvm_ffi.tensor -> !tvm_ffi.any
  return %alias, %alias, %input : !tvm_ffi.any, !tvm_ffi.any, !tvm_ffi.tensor
}

// Any-to-CAny conversion rewrites an owned call result and its exit.
// Newly generated Base calls must not acquire a second defer.
// DEFER-LABEL: func.func @any_call()
// DEFER-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @empty()
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: return %[[OK]], %[[VALUE]]
// CHECK-LABEL: func.func @any_call()
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @empty()
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[OK]], %[[VALUE]]
func.func @any_call() -> (!tvm_ffi.bool, !tvm_ffi.any) {
  %ok, %value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
  return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
}

// An Opaque call result gets automatic cleanup even with a Base function handle.
// Its Opaque exit retains an independent reference.
// DEFER-LABEL: func.func @base_handle_call(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
// CHECK-LABEL: func.func @base_handle_call(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
func.func @base_handle_call(%handle: !tvm_ffi.c.function) -> !tvm_ffi.any {
  %value = tvm_ffi.FunctionCall %handle : !tvm_ffi.c.function() : () -> !tvm_ffi.any
  return %value : !tvm_ffi.any
}

// Base extraction and branch results keep explicit ownership transfers.
// CHECK-LABEL: func.func @base_results(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK-NEXT: %[[SELECTED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi.c.tensor)
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: scf.yield %[[ELEMENT]] : !tvm_ffi.c.tensor
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK-NEXT: scf.yield %[[FALLBACK]] : !tvm_ffi.c.tensor
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SELECTED]] : !tvm_ffi.c.tensor
func.func @base_results(%tuple: !tvm_ffi.tuple<!tvm_ffi.c.tensor>, %fallback: !tvm_ffi.c.tensor, %ready: i1) -> !tvm_ffi.c.tensor {
  %selected = scf.if %ready -> (!tvm_ffi.c.tensor) {
    %element = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.c.tensor>) -> !tvm_ffi.c.tensor
    scf.yield %element : !tvm_ffi.c.tensor
  } else {
    tvm_ffi.ObjectIncRef %fallback : !tvm_ffi.c.tensor
    scf.yield %fallback : !tvm_ffi.c.tensor
  }
  return %selected : !tvm_ffi.c.tensor
}

// The conversion driver must apply the ownership pattern before folding a
// constant-condition if. Both branch exits export independent references.
// DEFER-LABEL: func.func @constant_if(
// DEFER-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any)
// DEFER: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.any
// DEFER-NEXT: scf.yield %[[FIRST]] : !tvm_ffi.c.any
// DEFER-NEXT: } else {
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]] : !tvm_ffi.c.any
// DEFER-NEXT: scf.yield %[[SECOND]] : !tvm_ffi.c.any
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[CHOSEN]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[CHOSEN]] : !tvm_ffi.c.any
// DEFER-NEXT: return %[[CHOSEN]]
// CHECK-LABEL: func.func @constant_if(
// CHECK: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if
// CHECK: tvm_ffi.ObjectIncRef %[[CHOSEN]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CHOSEN]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[CHOSEN]]
func.func @constant_if(%first: !tvm_ffi.any, %second: !tvm_ffi.any) -> !tvm_ffi.any {
  %condition = arith.constant true
  %chosen = scf.if %condition -> (!tvm_ffi.any) {
    scf.yield %first : !tvm_ffi.any
  } else {
    scf.yield %second : !tvm_ffi.any
  }
  return %chosen : !tvm_ffi.any
}

// Handwritten Base values keep their existing explicit lifetime management.
// CHECK-LABEL: func.func @manual(
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.exception
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.exception
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.exception
func.func @manual() -> !tvm_ffi.c.exception {
  %value = tvm_ffi.exception "GuardMatch" : !tvm_ffi.c.exception
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.c.exception
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.c.exception
  return %value : !tvm_ffi.c.exception
}

// A Base call and its Base result remain manually managed.
// DEFER-LABEL: func.func @manual_registry(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: return %[[VALUE]] : !tvm_ffi.c.tensor
// CHECK-LABEL: func.func @manual_registry(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.c.tensor
func.func @manual_registry(%handle: !tvm_ffi.c.function) -> !tvm_ffi.c.tensor {
  %value = tvm_ffi.FunctionCall %handle : !tvm_ffi.c.function() : () -> !tvm_ffi.c.tensor
  return %value : !tvm_ffi.c.tensor
}

// The operation replacement overload registers only the original Opaque
// result, even when both results have the same converted Base representation.
// DEFER-LABEL: func.func @mixed_if(
// DEFER-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[OWNED:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[MANUAL:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// DEFER-NEXT: %[[JOIN:[a-zA-Z0-9_]+]]:2 = scf.if %[[COND]]
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[OWNED]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[OWNED]], %[[MANUAL]]
// DEFER-NEXT: } else {
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[OWNED]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[OWNED]], %[[MANUAL]]
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// DEFER-NEXT: return %[[JOIN]]#0, %[[JOIN]]#1
// CHECK-LABEL: func.func @mixed_if(
// CHECK: %[[JOIN:[a-zA-Z0-9_]+]]:2 = scf.if
// CHECK: tvm_ffi.ObjectIncRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[JOIN]]#0, %[[JOIN]]#1
func.func @mixed_if(%condition: i1, %owned: !tvm_ffi.tensor, %manual: !tvm_ffi.c.tensor) -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor) {
  %semantic, %base = scf.if %condition -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor) {
    scf.yield %owned, %manual : !tvm_ffi.tensor, !tvm_ffi.c.tensor
  } else {
    scf.yield %owned, %manual : !tvm_ffi.tensor, !tvm_ffi.c.tensor
  }
  return %semantic, %base : !tvm_ffi.tensor, !tvm_ffi.c.tensor
}

// Opaque and Base tensor results have the same converted representation, but
// only the original Opaque results receive automatic local releases. Exports
// retain each Opaque occurrence before any local release, including duplicates.
// DEFER-LABEL: func.func @mixed_results(
// DEFER-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array)
// DEFER: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FIRST_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[FIRST_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[MANUAL_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[MANUAL_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[MANUAL_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[SCALAR_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SCALAR_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[SCALAR_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[LAST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[LAST_HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[LAST_INDEX:[a-zA-Z0-9_]+]])
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[LAST_HANDLE]] : !tvm_ffi.c.function
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[MANUAL]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[LAST]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// DEFER-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
// CHECK-LABEL: func.func @mixed_results(
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CHECK: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CHECK: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CHECK: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall
// CHECK: tvm_ffi.ObjectDecRef %[[MANUAL]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[LAST]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LAST]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
func.func @mixed_results(%tuple: !tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any>) -> (!tvm_ffi.tensor, !tvm_ffi.any, !tvm_ffi.tensor, !tvm_ffi.int) {
  %first, %manual, %scalar, %last = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any>) -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.int, !tvm_ffi.any)
  tvm_ffi.ObjectDecRef %manual : !tvm_ffi.c.tensor
  return %first, %last, %first, %scalar : !tvm_ffi.tensor, !tvm_ffi.any, !tvm_ffi.tensor, !tvm_ffi.int
}

// The shared Array/Tuple construction protocol borrows its elements. Temporary
// runtime handles are released immediately; containers are released in reverse
// construction order after all exports. Semantic strings keep their tagged
// cleanup contract even when the underlying constant uses static storage.
// CHECK-LABEL: func.func @nested_construction()
// CHECK-NEXT: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "text"
// CHECK-NEXT: %[[TEXT:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW]]
// CHECK-NEXT: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ARRAY_HANDLE]] : !tvm_ffi.c.function(%[[TEXT]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[TUPLE_HANDLE]] : !tvm_ffi.c.function(%[[ARRAY]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[TUPLE]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[TEXT]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TUPLE]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ARRAY]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TEXT]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[TUPLE]], %[[TEXT]]
func.func @nested_construction() -> (!tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>, !tvm_ffi.str) {
  %text = tvm_ffi.constant.str "text"
  %array = tvm_ffi.array.create(%text, %text) : (!tvm_ffi.str, !tvm_ffi.str) -> !tvm_ffi.array<!tvm_ffi.str>
  %tuple = tvm_ffi.tuple.construct(%array, %text) : (!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str) -> !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>
  return %tuple, %text : !tvm_ffi.tuple<!tvm_ffi.array<!tvm_ffi.str>, !tvm_ffi.str>, !tvm_ffi.str
}

// Nested Opaque branches transfer independent references at every yield.
// Cast chains borrow their input and do not register additional releases.
// DEFER-LABEL: func.func @nested_opaque(
// DEFER-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.array, %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// DEFER: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if %[[COND]]
// DEFER: %[[INNER:[a-zA-Z0-9_]+]] = scf.if %[[COND]]
// DEFER: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]] : !tvm_ffi.c.function(%[[TUPLE]], %[[OFFSET:[a-zA-Z0-9_]+]])
// DEFER: tvm_ffi.ObjectIncRef %[[ELEMENT]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[ELEMENT]]
// DEFER: tvm_ffi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[FALLBACK]]
// DEFER: tvm_ffi.ObjectIncRef %[[INNER]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[INNER]]
// DEFER: tvm_ffi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi.c.tensor
// DEFER-NEXT: scf.yield %[[FALLBACK]]
// DEFER: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[OUTER]]
// DEFER-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.c.tensor
// DEFER-NEXT: return %[[WIDE]], %[[NARROW]]
// CHECK-LABEL: func.func @nested_opaque(
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[OUTER]]
// CHECK-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.c.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTER]] : !tvm_ffi.c.tensor
// CHECK-NEXT: return %[[WIDE]], %[[NARROW]]
func.func @nested_opaque(%condition: i1, %tuple: !tvm_ffi.tuple<!tvm_ffi.tensor>, %fallback: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.tensor) {
  %outer = scf.if %condition -> (!tvm_ffi.tensor) {
    %inner = scf.if %condition -> (!tvm_ffi.tensor) {
      %element = tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<!tvm_ffi.tensor>) -> !tvm_ffi.tensor
      scf.yield %element : !tvm_ffi.tensor
    } else {
      scf.yield %fallback : !tvm_ffi.tensor
    }
    scf.yield %inner : !tvm_ffi.tensor
  } else {
    scf.yield %fallback : !tvm_ffi.tensor
  }
  %wide = tvm_ffi.upcast %outer : !tvm_ffi.tensor -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.tensor
  return %wide, %narrow : !tvm_ffi.any, !tvm_ffi.tensor
}

// Result containers borrow repeated payloads, acquiring no tensor cleanup.
// CHECK-LABEL: func.func @pure_borrows(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: %[[FIRST_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CHECK: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FIRST_HANDLE]] : !tvm_ffi.c.function(%[[FIRST_OK:[a-zA-Z0-9_]+]], %[[FIRST_VALUE]])
// CHECK: %[[SECOND_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CHECK: %[[SECOND_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[SECOND_HANDLE]] : !tvm_ffi.c.function(%[[SECOND_OK:[a-zA-Z0-9_]+]], %[[SECOND_VALUE]])
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND_HANDLE]] : !tvm_ffi.c.function
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.array
// CHECK-NEXT: return
func.func @pure_borrows(%input: !tvm_ffi.tensor) {
  %results:2 = tvm_ffi.monad.pure %input, %input : (!tvm_ffi.tensor, !tvm_ffi.tensor) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>)
  return
}

// Both successful and failed Result branches export independent references.
// CHECK-LABEL: func.func @result_exports(
// CHECK: %[[RESULTS:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY:[a-zA-Z0-9_]+]]
// CHECK: tvm_ffi.ObjectIncRef %[[SUCCESS:[a-zA-Z0-9_]+]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SUCCESS]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SUCCESS]] : !tvm_ffi.c.array
// CHECK-NEXT: scf.yield %[[SUCCESS]], %[[SUCCESS]]
// CHECK-NEXT: } else {
// CHECK: tvm_ffi.ObjectIncRef %[[FAILURE_FIRST:[a-zA-Z0-9_]+]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FAILURE_SECOND:[a-zA-Z0-9_]+]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FAILURE_SECOND]] : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FAILURE_FIRST]] : !tvm_ffi.c.array
// CHECK-NEXT: scf.yield %[[FAILURE_FIRST]], %[[FAILURE_SECOND]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULTS]]#0 : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULTS]]#1 : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULTS]]#1 : !tvm_ffi.c.array
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULTS]]#0 : !tvm_ffi.c.array
// CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1
func.func @result_exports(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>) {
  %outputs:2 = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (!tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>) {
  ^bb0(%payload: !tvm_ffi.tensor):
    %result = tvm_ffi.monad.pure %payload : (!tvm_ffi.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
    tvm_ffi.monad.yield %result, %result : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>
  }
  return %outputs#0, %outputs#1 : !tvm_ffi.monad.result<!tvm_ffi.tensor>, !tvm_ffi.monad.result<!tvm_ffi.tensor>
}

// A semantic call directly registers its Opaque result cleanup, even when
// the replacement callee and Union representation become Base.
// DEFER-LABEL: func.func @scalar_union(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// DEFER-NEXT: return %[[VALUE]]
// CHECK-LABEL: func.func @scalar_union(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CHECK-NEXT: return %[[VALUE]]
func.func @scalar_union(%handle: !tvm_ffi.function) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %value = tvm_ffi.FunctionCall %handle : !tvm_ffi.function() : () -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// An empty extraction produces no runtime call, result, or defer.
// DEFER-LABEL: func.func @zero_results(
// DEFER-NEXT: return
// CHECK-LABEL: func.func @zero_results(
// CHECK-NEXT: return
func.func @zero_results(%tuple: !tvm_ffi.tuple<>) {
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<>) -> ()
  return
}
