// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --verify-each | FileCheck %s --check-prefix=DEFER
// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --check-prefix=CLEANUP --implicit-check-not='tvm_ffi.defer'

tvm_ffi.func @empty() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi}

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
// CLEANUP-LABEL: func.func @any_call()
// CLEANUP-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.call @empty()
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: return %[[OK]], %[[VALUE]]
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
// CLEANUP-LABEL: func.func @base_handle_call(
// CLEANUP-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CLEANUP-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: return %[[VALUE]] : !tvm_ffi.c.any
func.func @base_handle_call(%handle: !tvm_ffi.c.function) -> !tvm_ffi.any {
  %value = tvm_ffi.FunctionCall %handle : !tvm_ffi.c.function() : () -> !tvm_ffi.any
  return %value : !tvm_ffi.any
}

// Borrowing patterns use the ordinary replacement helpers even when their
// original results are Opaque. Neither cast acquires a local reference.
// DEFER-LABEL: func.func @borrowed_casts(
// DEFER-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// DEFER-NEXT: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// DEFER-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.c.tensor
// DEFER-NEXT: return %[[WIDE]], %[[NARROW]]
// CLEANUP-LABEL: func.func @borrowed_casts(
// CLEANUP-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CLEANUP-NEXT: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[INPUT]]
// CLEANUP-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: return %[[WIDE]], %[[NARROW]]
func.func @borrowed_casts(%input: !tvm_ffi.tensor) -> (!tvm_ffi.any, !tvm_ffi.tensor) {
  %wide = tvm_ffi.upcast %input : !tvm_ffi.tensor -> !tvm_ffi.any
  %narrow = tvm_ffi.downcast %wide : !tvm_ffi.any -> !tvm_ffi.tensor
  return %wide, %narrow : !tvm_ffi.any, !tvm_ffi.tensor
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
// CLEANUP-LABEL: func.func @constant_if(
// CLEANUP: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if
// CLEANUP: tvm_ffi.ObjectIncRef %[[CHOSEN]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[CHOSEN]] : !tvm_ffi.c.any
// CLEANUP-NEXT: return %[[CHOSEN]]
func.func @constant_if(%first: !tvm_ffi.any, %second: !tvm_ffi.any) -> !tvm_ffi.any {
  %condition = arith.constant true
  %chosen = scf.if %condition -> (!tvm_ffi.any) {
    scf.yield %first : !tvm_ffi.any
  } else {
    scf.yield %second : !tvm_ffi.any
  }
  return %chosen : !tvm_ffi.any
}

// A Base call and its Base result remain manually managed.
// DEFER-LABEL: func.func @manual_registry(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: return %[[VALUE]] : !tvm_ffi.c.tensor
// CLEANUP-LABEL: func.func @manual_registry(
// CLEANUP-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CLEANUP-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CLEANUP-NEXT: return %[[VALUE]] : !tvm_ffi.c.tensor
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
// CLEANUP-LABEL: func.func @mixed_if(
// CLEANUP: %[[JOIN:[a-zA-Z0-9_]+]]:2 = scf.if
// CLEANUP: tvm_ffi.ObjectIncRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[JOIN]]#0 : !tvm_ffi.c.tensor
// CLEANUP-NEXT: return %[[JOIN]]#0, %[[JOIN]]#1
func.func @mixed_if(%condition: i1, %owned: !tvm_ffi.tensor, %manual: !tvm_ffi.c.tensor) -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor) {
  %semantic, %base = scf.if %condition -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor) {
    scf.yield %owned, %manual : !tvm_ffi.tensor, !tvm_ffi.c.tensor
  } else {
    scf.yield %owned, %manual : !tvm_ffi.tensor, !tvm_ffi.c.tensor
  }
  return %semantic, %base : !tvm_ffi.tensor, !tvm_ffi.c.tensor
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
// CLEANUP-LABEL: func.func @nested_opaque(
// CLEANUP: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if
// CLEANUP: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[OUTER]]
// CLEANUP-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[WIDE]]
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[OUTER]] : !tvm_ffi.c.tensor
// CLEANUP-NEXT: return %[[WIDE]], %[[NARROW]]
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
// CLEANUP-LABEL: func.func @scalar_union(
// CLEANUP-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.function)
// CLEANUP-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]
// CLEANUP-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.c.any
// CLEANUP-NEXT: return %[[VALUE]]
func.func @scalar_union(%handle: !tvm_ffi.function) -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
  %value = tvm_ffi.FunctionCall %handle : !tvm_ffi.function() : () -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
  return %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
}

// Newly constructed replacements use the original producer ownership, even
// when the replacement itself is a borrowing upcast. Each registers once.
// DEFER-LABEL: func.func @string_producers()
// DEFER-NEXT: %[[RAW_FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "first"
// DEFER-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW_FIRST]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: %[[RAW_SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "second"
// DEFER-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW_SECOND]]
// DEFER-NEXT: tvm_ffi.defer {
// DEFER-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.any
// DEFER-NEXT: tvm_ffi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: return
// CLEANUP-LABEL: func.func @string_producers()
// CLEANUP-NEXT: %[[RAW_FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "first"
// CLEANUP-NEXT: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW_FIRST]]
// CLEANUP-NEXT: %[[RAW_SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "second"
// CLEANUP-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[RAW_SECOND]]
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]] : !tvm_ffi.c.any
// CLEANUP-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]] : !tvm_ffi.c.any
// CLEANUP-NEXT: return
func.func @string_producers() {
  %first = tvm_ffi.constant.str "first"
  %second = tvm_ffi.constant.str "second"
  return
}

// An empty extraction produces no runtime call, result, or defer.
// DEFER-LABEL: func.func @zero_results(
// DEFER-NEXT: return
// CLEANUP-LABEL: func.func @zero_results(
// CLEANUP-NEXT: return
func.func @zero_results(%tuple: !tvm_ffi.tuple<>) {
  tvm_ffi.tuple.unconstruct %tuple : (!tvm_ffi.tuple<>) -> ()
  return
}
