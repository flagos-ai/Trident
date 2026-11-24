// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-tvm-ffi-api-to-abi --verify-each | FileCheck %s --check-prefix=DEFER
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-tvm-ffi-api-to-abi --lower-tvm-ffi-abi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi_abi.defer'

tvm_ffi_api.func @empty() -> !tvm_ffi_abi.none

// DEFER-LABEL: func.func @aliases(
// DEFER-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[FIRST]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[SECOND]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER: return

// CHECK-LABEL: func.func @aliases(
// CHECK-SAME: %[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[BORROWED]]
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[BORROWED]]
// CHECK: %[[ALIAS:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[FIRST]]
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[ALIAS]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[ALIAS]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[BORROWED]] : !tvm_ffi_abi.tensor
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[SECOND]] : !tvm_ffi_abi.tensor
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[FIRST]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: return %[[ALIAS]], %[[ALIAS]], %[[BORROWED]]
func.func @aliases(%input: !tvm_ffi_api.tensor) -> (!tvm_ffi_api.any, !tvm_ffi_api.any, !tvm_ffi_api.tensor) {
  %first = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
  %second = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
  %alias = tvm_ffi_api.upcast %first : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
  return %alias, %alias, %input : !tvm_ffi_api.any, !tvm_ffi_api.any, !tvm_ffi_api.tensor
}

// Any-to-CAny conversion rewrites an owned call result and its exit.
// Newly generated Base calls must not acquire a second defer.
// DEFER-LABEL: func.func @any_call()
// DEFER-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.call @empty()
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: return %[[OK]], %[[VALUE]]
// CHECK-LABEL: func.func @any_call()
// CHECK-NEXT: %[[OK:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.call @empty()
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[OK]], %[[VALUE]]
func.func @any_call() -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any) {
  %ok, %value = tvm_ffi_api.call @empty() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
  return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// An Opaque call result gets automatic cleanup even with a Base function handle.
// Its Opaque exit retains an independent reference.
// DEFER-LABEL: func.func @base_handle_call(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: return %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-LABEL: func.func @base_handle_call(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi_abi.any
func.func @base_handle_call(%handle: !tvm_ffi_abi.function) -> !tvm_ffi_api.any {
  %value = tvm_ffi_api.FunctionCall %handle : !tvm_ffi_abi.function() : () -> !tvm_ffi_api.any
  return %value : !tvm_ffi_api.any
}

// Base extraction and branch results keep explicit ownership transfers.
// DEFER-LABEL: func.func @base_results(
// DEFER: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[HANDLE]] : !tvm_ffi_abi.function
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: scf.yield %[[ELEMENT]] : !tvm_ffi_abi.tensor
// CHECK-LABEL: func.func @base_results(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array, %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[READY:[a-zA-Z0-9_]+]]: i1)
// CHECK-NEXT: %[[SELECTED:[a-zA-Z0-9_]+]] = scf.if %[[READY]] -> (!tvm_ffi_abi.tensor)
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK-NEXT: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[INDEX:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[HANDLE]] : !tvm_ffi_abi.function
// CHECK-NEXT: scf.yield %[[ELEMENT]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: scf.yield %[[FALLBACK]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SELECTED]] : !tvm_ffi_abi.tensor
func.func @base_results(%tuple: !tvm_ffi_api.tuple<!tvm_ffi_abi.tensor>, %fallback: !tvm_ffi_abi.tensor, %ready: i1) -> !tvm_ffi_abi.tensor {
  %selected = scf.if %ready -> (!tvm_ffi_abi.tensor) {
    %element = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_abi.tensor>) -> !tvm_ffi_abi.tensor
    scf.yield %element : !tvm_ffi_abi.tensor
  } else {
    tvm_ffi_abi.ObjectIncRef %fallback : !tvm_ffi_abi.tensor
    scf.yield %fallback : !tvm_ffi_abi.tensor
  }
  return %selected : !tvm_ffi_abi.tensor
}

// CFG parameters borrow; the entry-owned reference stays alive across the join.
// DEFER-LABEL: func.func @cfg_borrow(
// DEFER-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor,
// DEFER: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]]
// DEFER: cf.cond_br
// DEFER: return
// CHECK-LABEL: func.func @cfg_borrow(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[COND:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// CHECK-NEXT: cf.cond_br %[[COND]], ^[[LEFT:bb[0-9]+]](%[[VALUE]] : !tvm_ffi_abi.tensor), ^[[JOIN:bb[0-9]+]](%[[VALUE]] : !tvm_ffi_abi.tensor)
// CHECK: ^[[LEFT]](%[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor):
// CHECK-NEXT: cf.br ^[[LEFT_BODY:bb[0-9]+]](%[[BORROWED]] : !tvm_ffi_abi.tensor)
// CHECK: ^[[LEFT_BODY]](%[[LEFT_VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor):
// CHECK-NEXT: cf.br ^[[LEFT_EDGE:bb[0-9]+]](%[[LEFT_VALUE]] : !tvm_ffi_abi.tensor)
// CHECK: ^[[JOIN]](%[[DIRECT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor):
// CHECK-NEXT: cf.br ^[[FINAL:bb[0-9]+]](%[[DIRECT]] : !tvm_ffi_abi.tensor)
// CHECK: ^[[LEFT_EDGE]](%[[FORWARDED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor):
// CHECK-NEXT: cf.br ^[[FINAL]](%[[FORWARDED]] : !tvm_ffi_abi.tensor)
// CHECK: ^[[FINAL]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor):
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[RESULT]]
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]]
// CHECK-NEXT: return %[[RESULT]]
func.func @cfg_borrow(%input: !tvm_ffi_api.tensor, %condition: i1) -> !tvm_ffi_api.tensor {
  %value = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
  cf.cond_br %condition, ^left(%value : !tvm_ffi_api.tensor), ^join(%value : !tvm_ffi_api.tensor)
^left(%borrowed: !tvm_ffi_api.tensor):
  cf.br ^join(%borrowed : !tvm_ffi_api.tensor)
^join(%result: !tvm_ffi_api.tensor):
  return %result : !tvm_ffi_api.tensor
}

// Packed function CFG exits each export before releasing the shared owner.
// DEFER-LABEL: tvm_ffi_abi.func @cfg_packed(
// DEFER: tvm_ffi_abi.defer {
// DEFER: cf.cond_br
// CHECK-LABEL: tvm_ffi_abi.func @cfg_packed(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[COND:[a-zA-Z0-9_]+]]: i1)
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.tensor.clone %[[INPUT]]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[CLONE]]
// CHECK: cf.cond_br %[[COND]], ^[[LEFT:bb[0-9]+]], ^[[RIGHT:bb[0-9]+]]
// CHECK: ^[[LEFT]]:
// CHECK-NEXT: cf.br ^[[LEFT_BODY:bb[0-9]+]]
// CHECK: ^[[LEFT_BODY]]:
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[PAYLOAD]]
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[CLONE]]
// CHECK-NEXT: tvm_ffi_abi.return %[[OK:[a-zA-Z0-9_]+]], %[[PAYLOAD]]
// CHECK: ^[[RIGHT]]:
// CHECK-NEXT: cf.br ^[[RIGHT_BODY:bb[0-9]+]]
// CHECK: ^[[RIGHT_BODY]]:
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[PAYLOAD]]
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[CLONE]]
// CHECK-NEXT: tvm_ffi_abi.return %[[OK]], %[[PAYLOAD]]
tvm_ffi_api.func @cfg_packed(%input: !tvm_ffi_api.tensor, %condition: i1) -> !tvm_ffi_api.tensor {
  %clone = tvm_ffi_api.tensor.clone %input : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
  %payload = tvm_ffi_api.upcast %clone : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
  %ok = tvm_ffi_abi.constant.bool true
  cf.cond_br %condition, ^left, ^right
^left:
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
^right:
  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
}

// Both exits of a constant-condition if export independent references.
// DEFER-LABEL: func.func @constant_if(
// DEFER-SAME: %[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any, %[[SECOND:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.any)
// DEFER: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[FIRST]] : !tvm_ffi_abi.any
// DEFER-NEXT: scf.yield %[[FIRST]] : !tvm_ffi_abi.any
// DEFER-NEXT: } else {
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[SECOND]] : !tvm_ffi_abi.any
// DEFER-NEXT: scf.yield %[[SECOND]] : !tvm_ffi_abi.any
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[CHOSEN]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[CHOSEN]] : !tvm_ffi_abi.any
// DEFER-NEXT: return %[[CHOSEN]]
// CHECK-LABEL: func.func @constant_if(
// CHECK: %[[CHOSEN:[a-zA-Z0-9_]+]] = scf.if
// CHECK: tvm_ffi_abi.ObjectIncRef %[[CHOSEN]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[CHOSEN]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[CHOSEN]]
func.func @constant_if(%first: !tvm_ffi_api.any, %second: !tvm_ffi_api.any) -> !tvm_ffi_api.any {
  %condition = arith.constant true
  %chosen = scf.if %condition -> (!tvm_ffi_api.any) {
    scf.yield %first : !tvm_ffi_api.any
  } else {
    scf.yield %second : !tvm_ffi_api.any
  }
  return %chosen : !tvm_ffi_api.any
}

// Handwritten Base values keep their existing explicit lifetime management.
// CHECK-LABEL: func.func @manual(
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.exception "GuardMatch" : !tvm_ffi_abi.exception
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.exception
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.exception
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi_abi.exception
func.func @manual() -> !tvm_ffi_abi.exception {
  %value = tvm_ffi_api.exception "GuardMatch" : !tvm_ffi_abi.exception
  tvm_ffi_abi.ObjectIncRef %value : !tvm_ffi_abi.exception
  tvm_ffi_abi.ObjectDecRef %value : !tvm_ffi_abi.exception
  return %value : !tvm_ffi_abi.exception
}

// A Base call and its Base result remain manually managed.
// DEFER-LABEL: func.func @manual_registry(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: return %[[VALUE]] : !tvm_ffi_abi.tensor
// CHECK-LABEL: func.func @manual_registry(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi_abi.tensor
func.func @manual_registry(%handle: !tvm_ffi_abi.function) -> !tvm_ffi_abi.tensor {
  %value = tvm_ffi_abi.FunctionCall %handle : !tvm_ffi_abi.function() : () -> !tvm_ffi_abi.tensor
  return %value : !tvm_ffi_abi.tensor
}

// The operation replacement overload registers only the original Opaque
// result, even when both results have the same converted Base representation.
// DEFER-LABEL: func.func @mixed_if(
// DEFER-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[OWNED:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor, %[[MANUAL:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// DEFER-NEXT: %[[JOIN:[a-zA-Z0-9_]+]]:2 = scf.if %[[COND]]
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[OWNED]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[OWNED]], %[[MANUAL]]
// DEFER-NEXT: } else {
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[OWNED]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[OWNED]], %[[MANUAL]]
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[JOIN]]#0 : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[JOIN]]#0 : !tvm_ffi_abi.tensor
// DEFER-NEXT: return %[[JOIN]]#0, %[[JOIN]]#1
// CHECK-LABEL: func.func @mixed_if(
// CHECK: %[[JOIN:[a-zA-Z0-9_]+]]:2 = scf.if
// CHECK: tvm_ffi_abi.ObjectIncRef %[[JOIN]]#0 : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[JOIN]]#0 : !tvm_ffi_abi.tensor
// CHECK-NEXT: return %[[JOIN]]#0, %[[JOIN]]#1
func.func @mixed_if(%condition: i1, %owned: !tvm_ffi_api.tensor, %manual: !tvm_ffi_abi.tensor) -> (!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor) {
  %semantic, %base = scf.if %condition -> (!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor) {
    scf.yield %owned, %manual : !tvm_ffi_api.tensor, !tvm_ffi_abi.tensor
  } else {
    scf.yield %owned, %manual : !tvm_ffi_api.tensor, !tvm_ffi_abi.tensor
  }
  return %semantic, %base : !tvm_ffi_api.tensor, !tvm_ffi_abi.tensor
}

// Opaque and Base tensor results have the same converted representation, but
// only the original Opaque results receive automatic local releases. Exports
// retain each Opaque occurrence before any local release, including duplicates.
// DEFER-LABEL: func.func @mixed_results(
// DEFER-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array)
// DEFER: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[FIRST_HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[FIRST_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[MANUAL_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[MANUAL_HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[MANUAL_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[SCALAR_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[SCALAR_HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[SCALAR_INDEX:[a-zA-Z0-9_]+]])
// DEFER: %[[LAST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[LAST_HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[LAST_INDEX:[a-zA-Z0-9_]+]])
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[LAST_HANDLE]] : !tvm_ffi_abi.function
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[FIRST]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[LAST]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[MANUAL]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[FIRST]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[LAST]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[FIRST]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
// CHECK-LABEL: func.func @mixed_results(
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK: %[[MANUAL:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK: %[[SCALAR:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK: tvm_ffi_abi.ObjectDecRef %[[MANUAL]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[FIRST]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[LAST]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[FIRST]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[FIRST]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[LAST]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[FIRST]], %[[LAST]], %[[FIRST]], %[[SCALAR]]
func.func @mixed_results(%tuple: !tvm_ffi_api.tuple<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.int, !tvm_ffi_api.any>) -> (!tvm_ffi_api.tensor, !tvm_ffi_api.any, !tvm_ffi_api.tensor, !tvm_ffi_abi.int) {
  %first, %manual, %scalar, %last = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.int, !tvm_ffi_api.any>) -> (!tvm_ffi_api.tensor, !tvm_ffi_abi.tensor, !tvm_ffi_abi.int, !tvm_ffi_api.any)
  tvm_ffi_abi.ObjectDecRef %manual : !tvm_ffi_abi.tensor
  return %first, %last, %first, %scalar : !tvm_ffi_api.tensor, !tvm_ffi_api.any, !tvm_ffi_api.tensor, !tvm_ffi_abi.int
}

// The shared Array/Tuple construction protocol borrows its elements. Temporary
// runtime handles are released immediately; containers are released after all
// exports without prescribing their relative order. Semantic strings keep their tagged
// cleanup contract even when the underlying constant uses static storage.
// CHECK-LABEL: func.func @nested_construction()
// CHECK-NEXT: %[[RAW:[a-zA-Z0-9_]+]] = tvm_ffi_abi.constant.raw_str "text"
// CHECK-NEXT: %[[TEXT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[RAW]]
// CHECK-NEXT: %[[ARRAY_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[ARRAY:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[ARRAY_HANDLE]] : !tvm_ffi_abi.function(%[[TEXT]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[ARRAY_HANDLE]] : !tvm_ffi_abi.function
// CHECK-NEXT: %[[TUPLE_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.Array"
// CHECK-NEXT: %[[TUPLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[TUPLE_HANDLE]] : !tvm_ffi_abi.function(%[[ARRAY]], %[[TEXT]])
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[TUPLE_HANDLE]] : !tvm_ffi_abi.function
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[TUPLE]] : !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[TEXT]] : !tvm_ffi_abi.any
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[TUPLE]] : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[ARRAY]] : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[TEXT]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[TUPLE]], %[[TEXT]]
func.func @nested_construction() -> (!tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.str>, !tvm_ffi_api.str>, !tvm_ffi_api.str) {
  %text = tvm_ffi_api.constant.str "text"
  %array = tvm_ffi_api.array.create(%text, %text) : (!tvm_ffi_api.str, !tvm_ffi_api.str) -> !tvm_ffi_api.array<!tvm_ffi_api.str>
  %tuple = tvm_ffi_api.tuple.construct(%array, %text) : (!tvm_ffi_api.array<!tvm_ffi_api.str>, !tvm_ffi_api.str) -> !tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.str>, !tvm_ffi_api.str>
  return %tuple, %text : !tvm_ffi_api.tuple<!tvm_ffi_api.array<!tvm_ffi_api.str>, !tvm_ffi_api.str>, !tvm_ffi_api.str
}

// Nested Opaque branches transfer independent references at every yield.
// Cast chains borrow their input and do not register additional releases.
// DEFER-LABEL: func.func @nested_opaque(
// DEFER-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.array, %[[FALLBACK:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// DEFER: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if %[[COND]]
// DEFER: %[[INNER:[a-zA-Z0-9_]+]] = scf.if %[[COND]]
// DEFER: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// DEFER: %[[ELEMENT:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]] : !tvm_ffi_abi.function(%[[TUPLE]], %[[OFFSET:[a-zA-Z0-9_]+]])
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[HANDLE]] : !tvm_ffi_abi.function
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[ELEMENT]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[ELEMENT]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[ELEMENT]]
// DEFER: tvm_ffi_abi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[FALLBACK]]
// DEFER: tvm_ffi_abi.ObjectIncRef %[[INNER]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[INNER]]
// DEFER: tvm_ffi_abi.ObjectIncRef %[[FALLBACK]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: scf.yield %[[FALLBACK]]
// DEFER: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[OUTER]]
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[WIDE]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[OUTER]] : !tvm_ffi_abi.tensor
// DEFER-NEXT: return %[[WIDE]], %[[OUTER]]
// CHECK-LABEL: func.func @nested_opaque(
// CHECK: %[[OUTER:[a-zA-Z0-9_]+]] = scf.if
// CHECK: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[OUTER]]
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[WIDE]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[OUTER]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[OUTER]] : !tvm_ffi_abi.tensor
// CHECK-NEXT: return %[[WIDE]], %[[OUTER]]
func.func @nested_opaque(%condition: i1, %tuple: !tvm_ffi_api.tuple<!tvm_ffi_api.tensor>, %fallback: !tvm_ffi_api.tensor) -> (!tvm_ffi_api.any, !tvm_ffi_api.tensor) {
  %outer = scf.if %condition -> (!tvm_ffi_api.tensor) {
    %inner = scf.if %condition -> (!tvm_ffi_api.tensor) {
      %element = tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<!tvm_ffi_api.tensor>) -> !tvm_ffi_api.tensor
      scf.yield %element : !tvm_ffi_api.tensor
    } else {
      scf.yield %fallback : !tvm_ffi_api.tensor
    }
    scf.yield %inner : !tvm_ffi_api.tensor
  } else {
    scf.yield %fallback : !tvm_ffi_api.tensor
  }
  %wide = tvm_ffi_api.upcast %outer : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
  %narrow = tvm_ffi_api.downcast %wide : !tvm_ffi_api.any -> !tvm_ffi_api.tensor
  return %wide, %narrow : !tvm_ffi_api.any, !tvm_ffi_api.tensor
}

// Result containers borrow repeated payloads, acquiring no tensor cleanup.
// CHECK-LABEL: func.func @pure_borrows(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.tensor)
// CHECK: %[[FIRST_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[INPUT]]
// CHECK: %[[FIRST_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[FIRST_HANDLE]] : !tvm_ffi_abi.function(%[[FIRST_OK:[a-zA-Z0-9_]+]], %[[FIRST_VALUE]])
// CHECK: %[[SECOND_VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.upcast %[[INPUT]]
// CHECK: %[[SECOND_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.Array"
// CHECK: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[SECOND_HANDLE]] : !tvm_ffi_abi.function(%[[SECOND_OK:[a-zA-Z0-9_]+]], %[[SECOND_VALUE]])
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[SECOND_HANDLE]] : !tvm_ffi_abi.function
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[SECOND]] : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[FIRST]] : !tvm_ffi_abi.array
// CHECK-NEXT: return
func.func @pure_borrows(%input: !tvm_ffi_api.tensor) {
  %results_0 = tvm_ffi_monad.pure %input : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  %results_1 = tvm_ffi_monad.pure %input : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  return
}

// Both successful and failed Result branches export independent references.
// CHECK-LABEL: func.func @result_exports(
// CHECK: %[[STATUS_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[PAYLOAD_HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionGetGlobal "ffi.ArrayGetItem"
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[PAYLOAD_HANDLE]]
// CHECK: %[[RESULTS:[a-zA-Z0-9_]+]]:2 = scf.if %[[READY:[a-zA-Z0-9_]+]]
// CHECK: tvm_ffi_abi.ObjectIncRef %[[SUCCESS:[a-zA-Z0-9_]+]] : !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[SUCCESS]] : !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[SUCCESS]] : !tvm_ffi_abi.array
// CHECK-NEXT: scf.yield %[[SUCCESS]], %[[SUCCESS]]
// CHECK-NEXT: } else {
// CHECK: %[[FAILURE_FIRST:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK: %[[FAILURE_SECOND:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[HANDLE:[a-zA-Z0-9_]+]] : !tvm_ffi_abi.function
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[FAILURE_FIRST]] : !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[FAILURE_SECOND]] : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[FAILURE_SECOND]] : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[FAILURE_FIRST]] : !tvm_ffi_abi.array
// CHECK-NEXT: scf.yield %[[FAILURE_FIRST]], %[[FAILURE_SECOND]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[RESULTS]]#0 : !tvm_ffi_abi.array
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[RESULTS]]#1 : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[PAYLOAD]] : !tvm_ffi_abi.any
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[RESULTS]]#0 : !tvm_ffi_abi.array
// CHECK-DAG: tvm_ffi_abi.ObjectDecRef %[[RESULTS]]#1 : !tvm_ffi_abi.array
// CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1
func.func @result_exports(%input: !tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_api.tensor>) {
  %outputs:2 = tvm_ffi_monad.bind %input : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_api.tensor>) {
  ^bb0(%payload: !tvm_ffi_api.tensor):
    %result = tvm_ffi_monad.pure %payload : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
    tvm_ffi_monad.yield %result, %result : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  }
  return %outputs#0, %outputs#1 : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>, !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
}

// A semantic call directly registers its Opaque result cleanup, even when
// the replacement callee and Union representation become Base.
// DEFER-LABEL: func.func @scalar_union(
// DEFER-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// DEFER-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// DEFER-NEXT: return %[[VALUE]]
// CHECK-LABEL: func.func @scalar_union(
// CHECK-SAME: %[[HANDLE:[a-zA-Z0-9_]+]]: !tvm_ffi_abi.function)
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi_abi.FunctionCall %[[HANDLE]]
// CHECK-NEXT: tvm_ffi_abi.ObjectIncRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: tvm_ffi_abi.ObjectDecRef %[[VALUE]] : !tvm_ffi_abi.any
// CHECK-NEXT: return %[[VALUE]]
func.func @scalar_union(%handle: !tvm_ffi_api.function) -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> {
  %value = tvm_ffi_api.FunctionCall %handle : !tvm_ffi_api.function() : () -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
  return %value : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>
}

// An empty extraction registers a terminator-only defer without runtime calls.
// DEFER-LABEL: func.func @zero_results(
// DEFER-NEXT: tvm_ffi_abi.defer {
// DEFER-NEXT: tvm_ffi_abi.defer_yield
// DEFER-NEXT: }
// DEFER-NEXT: return
// CHECK-LABEL: func.func @zero_results(
// CHECK-NEXT: return
func.func @zero_results(%tuple: !tvm_ffi_api.tuple<>) {
  tvm_ffi_api.tuple.unconstruct %tuple : (!tvm_ffi_api.tuple<>) -> ()
  return
}
