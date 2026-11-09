// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --decompose-tvm-ffi --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='!tvm_ffi.' --implicit-check-not='scf.' --implicit-check-not='unrealized_conversion_cast'

// Synchronous captures borrow the outer reference. Local producers are released
// at the nested region exit without a parent-operation ownership whitelist.
// CHECK-LABEL: func.func @execute_capture(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK-NEXT: scf.execute_region {
// CHECK-NEXT: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CLONE]]
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
// LLVM-LABEL: llvm.func @execute_capture(
func.func @execute_capture(%input: !tvm_ffi.tensor) {
  scf.execute_region {
    %clone = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
    scf.yield
  }
  return
}

// A converted loop entry argument flows through independently converted nested
// branch yields and the loop yield. Each boundary retains before local cleanup.
// CHECK-LABEL: func.func @for_nested_if(
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.for
// CHECK-SAME: iter_args(%[[CURRENT:[a-zA-Z0-9_]+]] = %[[INPUT:[a-zA-Z0-9_]+]])
// CHECK-NEXT: %[[NEXT:[a-zA-Z0-9_]+]] = scf.if
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[CURRENT]]
// CHECK-NEXT: scf.yield %[[CURRENT]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[CURRENT]]
// CHECK-NEXT: scf.yield %[[CURRENT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[NEXT]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[NEXT]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[CURRENT]]
// CHECK-NEXT: scf.yield %[[NEXT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]
// CHECK-NEXT: return %[[RESULT]]
// LLVM-LABEL: llvm.func @for_nested_if(
func.func @for_nested_if(%input: !tvm_ffi.tensor, %condition: i1, %lower: index, %upper: index, %step: index) -> !tvm_ffi.tensor {
  %result = scf.for %i = %lower to %upper step %step iter_args(%current = %input) -> (!tvm_ffi.tensor) {
    %next = scf.if %condition -> (!tvm_ffi.tensor) {
      scf.yield %current : !tvm_ffi.tensor
    } else {
      scf.yield %current : !tvm_ffi.tensor
    }
    scf.yield %next : !tvm_ffi.tensor
  }
  return %result : !tvm_ffi.tensor
}

// A dynamic trip count includes zero, one and many iterations. Each repeated
// semantic slot owns a reference; the identically represented Base slot borrows.
// CHECK-LABEL: func.func @for_slots(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor,
// CHECK: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]]:3 = scf.for unsigned
// CHECK-SAME: iter_args(%[[FIRST:[a-zA-Z0-9_]+]] = %[[INPUT]], %[[SECOND:[a-zA-Z0-9_]+]] = %[[INPUT]], %[[MANUAL:[a-zA-Z0-9_]+]] = %[[BASE:[a-zA-Z0-9_]+]])
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-NEXT: scf.yield %[[SECOND]], %[[FIRST]], %[[MANUAL]]
// CHECK-NEXT: } {test.marker = "preserved"}
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#0
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#1
// CHECK-NEXT: return
// LLVM-LABEL: llvm.func @for_slots(
func.func @for_slots(%input: !tvm_ffi.tensor, %manual: !tvm_ffi.c.tensor, %lower: index, %upper: index, %step: index) {
  %first, %second, %base = scf.for unsigned %i = %lower to %upper step %step iter_args(%a = %input, %b = %input, %c = %manual) -> (!tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.c.tensor) {
    scf.yield %b, %a, %c : !tvm_ffi.tensor, !tvm_ffi.tensor, !tvm_ffi.c.tensor
  } {test.marker = "preserved"}
  return
}

// Captures borrow on branch entry. Repeated exports each retain a reference,
// before releasing the producer reference.
// CHECK-LABEL: func.func @if_repeated_exports(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]]:2 = scf.if %[[COND]]
// CHECK-NEXT: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[CLONE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[CLONE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[CLONE]]
// CHECK-NEXT: scf.yield {test.marker = "if_yield"} %[[CLONE]], %[[CLONE]]
// CHECK-NEXT: } else {
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: scf.yield %[[INPUT]], %[[INPUT]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]]#0
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#0
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#1
// CHECK-NEXT: return %[[RESULT]]#0
// LLVM-LABEL: llvm.func @if_repeated_exports(
func.func @if_repeated_exports(%condition: i1, %input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %first, %second = scf.if %condition -> (!tvm_ffi.tensor, !tvm_ffi.tensor) {
    %clone = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
    scf.yield {test.marker = "if_yield"} %clone, %clone : !tvm_ffi.tensor, !tvm_ffi.tensor
  } else {
    scf.yield %input, %input : !tvm_ffi.tensor, !tvm_ffi.tensor
  }
  return %first : !tvm_ffi.tensor
}

// A cast is borrowed, so exporting it must retain before both its owner and
// the incoming reference are released. Before/after payload types may differ.
// CHECK-LABEL: func.func @while_cast(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[COND:[a-zA-Z0-9_]+]]: i1)
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.while (%[[BEFORE:[a-zA-Z0-9_]+]] = %[[INPUT]])
// CHECK-NEXT: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[BEFORE]]
// CHECK-NEXT: %[[WIDE:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[CLONE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[WIDE]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[CLONE]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[BEFORE]]
// CHECK-NEXT: scf.condition(%[[COND]]) %[[WIDE]]
// CHECK: ^[[BODY:[a-zA-Z0-9_]+]](%[[AFTER:[a-zA-Z0-9_]+]]: !tvm_ffi.c.any):
// CHECK-NEXT: %[[NARROW:[a-zA-Z0-9_]+]] = tvm_ffi.downcast %[[AFTER]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[NARROW]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[AFTER]]
// CHECK-NEXT: scf.yield %[[NARROW]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]
// CHECK-NEXT: return %[[RESULT]]
// LLVM-LABEL: llvm.func @while_cast(
func.func @while_cast(%input: !tvm_ffi.tensor, %condition: i1) -> !tvm_ffi.any {
  %result = scf.while (%before = %input) : (!tvm_ffi.tensor) -> !tvm_ffi.any {
    %clone = tvm_ffi.tensor.clone %before : !tvm_ffi.tensor -> !tvm_ffi.tensor
    %wide = tvm_ffi.upcast %clone : !tvm_ffi.tensor -> !tvm_ffi.any
    scf.condition(%condition) %wide : !tvm_ffi.any
  } do {
  ^body(%after: !tvm_ffi.any):
    %narrow = tvm_ffi.downcast %after : !tvm_ffi.any -> !tvm_ffi.tensor
    scf.yield %narrow : !tvm_ffi.tensor
  }
  return %result : !tvm_ffi.any
}

// Before and after regions have different arities. Repeated condition slots
// retain independently, while the Base slot receives no automatic references.
// CHECK-LABEL: func.func @while_slots(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[MANUAL:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[COND:[a-zA-Z0-9_]+]]: i1)
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]]
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]]:3 = scf.while (%[[BEFORE:[a-zA-Z0-9_]+]] = %[[INPUT]])
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BEFORE]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BEFORE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[BEFORE]]
// CHECK-NEXT: scf.condition(%[[COND]]) %[[BEFORE]], %[[MANUAL]], %[[BEFORE]]
// CHECK-NEXT: } do {
// CHECK-NEXT: ^[[BODY:[a-zA-Z0-9_]+]](%[[FIRST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[BASE:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[LAST:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor):
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[LAST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LAST]]
// CHECK-NEXT: scf.yield %[[LAST]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#0
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]]#2
// CHECK-NEXT: return
// LLVM-LABEL: llvm.func @while_slots(
func.func @while_slots(%input: !tvm_ffi.tensor, %manual: !tvm_ffi.c.tensor, %condition: i1) {
  %first, %base, %last = scf.while (%before = %input) : (!tvm_ffi.tensor) -> (!tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.tensor) {
    scf.condition(%condition) %before, %manual, %before : !tvm_ffi.tensor, !tvm_ffi.c.tensor, !tvm_ffi.tensor
  } do {
  ^body(%a: !tvm_ffi.tensor, %b: !tvm_ffi.c.tensor, %c: !tvm_ffi.tensor):
    scf.yield %c : !tvm_ffi.tensor
  }
  return
}
