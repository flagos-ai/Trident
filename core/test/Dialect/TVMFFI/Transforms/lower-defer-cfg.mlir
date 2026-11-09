// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-defer --lower-tvm-ffi-defer --verify-each | FileCheck %s --implicit-check-not='tvm_ffi.defer'

// Body-local SSA is cloned independently at each exit, retaining data flow.
// CHECK-LABEL: func.func @cfg_local_ssa(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: i1)
// CHECK: cf.cond_br %[[C]], ^[[LEFT:bb[0-9]+]], ^[[RIGHT:bb[0-9]+]]
// CHECK: ^[[LEFT]]:
// CHECK-NEXT: cf.br ^[[LEFT_BODY:bb[0-9]+]]
// CHECK: ^[[LEFT_BODY]]:
// CHECK: %[[A:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK-NEXT: %[[WA:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[A]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[WA]]
// CHECK-NEXT: return
// CHECK: ^[[RIGHT]]:
// CHECK-NEXT: cf.br ^[[RIGHT_BODY:bb[0-9]+]]
// CHECK: ^[[RIGHT_BODY]]:
// CHECK: %[[B:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK-NEXT: %[[WB:[a-zA-Z0-9_]+]] = tvm_ffi.upcast %[[B]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[WB]]
// CHECK-NEXT: return
func.func @cfg_local_ssa(%c: i1) {
  tvm_ffi.defer {
    %none = tvm_ffi.constant.none
    %wide = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.c.any
    tvm_ffi.ObjectDecRef %wide : !tvm_ffi.c.any
    tvm_ffi.defer_yield
  }
  cf.cond_br %c, ^left, ^right
^left:
  return
^right:
  return
}

// The left registration exits its scope on two distinct edges to the join.
// CHECK-LABEL: func.func @critical_edges(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: i1, %[[V:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[X:[a-zA-Z0-9_]+]]: i32, %[[Y:[a-zA-Z0-9_]+]]: i32)
// CHECK: cf.cond_br %[[C]], ^[[LEFT:bb[0-9]+]], ^[[JOIN:bb[0-9]+]](%[[X]] : i32)
// CHECK: ^[[LEFT]]:
// CHECK-NEXT: cf.cond_br %[[C]], ^[[E1:bb[0-9]+]](%[[X]] : i32), ^[[E2:bb[0-9]+]](%[[Y]] : i32)
// CHECK: ^[[E1]](%[[P1:[a-zA-Z0-9_]+]]: i32):
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[V]]
// CHECK-NEXT: cf.br ^[[JOIN]](%[[P1]] : i32)
// CHECK: ^[[E2]](%[[P2:[a-zA-Z0-9_]+]]: i32):
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[V]]
// CHECK-NEXT: cf.br ^[[JOIN]](%[[P2]] : i32)
// CHECK: ^[[JOIN]](%[[P:[a-zA-Z0-9_]+]]: i32):
// CHECK-NEXT: return %[[P]]
func.func @critical_edges(%c: i1, %v: !tvm_ffi.c.tensor, %x: i32, %y: i32) -> i32 {
  cf.cond_br %c, ^left, ^join(%x : i32)
^left:
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %v : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.cond_br %c, ^join(%x : i32), ^join(%y : i32)
^join(%p: i32):
  return %p : i32
}

// The loop header owner survives a body use, then cleans up on a latch backedge.
// CHECK-LABEL: func.func @latch(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: i1, %[[V:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: cf.br ^[[HEADER:bb[0-9]+]]
// CHECK: ^[[BACK:bb[0-9]+]]:
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[V]]
// CHECK-NEXT: cf.br ^[[HEADER]]
// CHECK: ^[[HEADER]]:
// CHECK-NEXT: cf.br ^[[FORWARD:bb[0-9]+]]
// CHECK: ^[[FORWARD]]:
// CHECK-NEXT: cf.br ^[[LATCH:bb[0-9]+]]
// CHECK: ^[[LATCH]]:
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[V]]
// CHECK-NEXT: cf.cond_br %[[C]], ^[[BACK]], ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: cf.br ^[[FINAL:bb[0-9]+]]
// CHECK: ^[[FINAL]]:
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[V]]
// CHECK-NEXT: return
func.func @latch(%c: i1, %v: !tvm_ffi.c.tensor) {
  cf.br ^header
^header:
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %v : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.br ^latch
^latch:
  tvm_ffi.ObjectIncRef %v : !tvm_ffi.c.tensor
  cf.cond_br %c, ^header, ^exit
^exit:
  return
}

// A loop registration runs on both the self-edge and exit; outer cleanup waits.
// CHECK-LABEL: func.func @loop(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: i1, %[[OUTER:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[INNER:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK: cf.br ^[[ENTRY_EDGE:bb[0-9]+]]
// CHECK: ^[[ENTRY_EDGE]]:
// CHECK-NEXT: cf.br ^[[BODY:bb[0-9]+]]
// CHECK: ^[[BACK:bb[0-9]+]]:
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK-NEXT: cf.br ^[[BODY]]
// CHECK: ^[[BODY]]:
// CHECK-NEXT: cf.cond_br %[[C]], ^[[BACK]], ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: cf.br ^[[FINAL:bb[0-9]+]]
// CHECK: ^[[FINAL]]:
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[INNER]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[OUTER]]
// CHECK-NEXT: return
func.func @loop(%c: i1, %outer: !tvm_ffi.c.tensor, %inner: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %outer : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.br ^body
^body:
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %inner : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.cond_br %c, ^body, ^exit
^exit:
  return
}

// Multiple registrations share exits without extending inner owners' scopes.
// CHECK-LABEL: func.func @scoped_exits(
// CHECK-SAME: %[[C:[a-zA-Z0-9_]+]]: i1, %[[A:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[B:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[X:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor, %[[Y:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK-NEXT: cf.cond_br %[[C]], ^[[INNER_EDGE:bb[0-9]+]], ^[[JOIN_EDGE:bb[0-9]+]]
// CHECK: ^[[INNER_EDGE]]:
// CHECK-NEXT: cf.br ^[[INNER:bb[0-9]+]]
// CHECK: ^[[INNER]]:
// CHECK-NEXT: cf.br ^[[EDGE:bb[0-9]+]]
// CHECK: ^[[JOIN_EDGE]]:
// CHECK-NEXT: cf.br ^[[JOIN:bb[0-9]+]]
// CHECK: ^[[EDGE]]:
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[X]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[Y]]
// CHECK-NEXT: cf.br ^[[JOIN]]
// CHECK: ^[[JOIN]]:
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[A]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[B]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[A]]
// CHECK-DAG: tvm_ffi.ObjectDecRef %[[B]]
// CHECK-NEXT: return
func.func @scoped_exits(%c: i1, %a: !tvm_ffi.c.tensor, %b: !tvm_ffi.c.tensor, %x: !tvm_ffi.c.tensor, %y: !tvm_ffi.c.tensor) {
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %a : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %b : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.cond_br %c, ^inner, ^join
^inner:
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %x : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %y : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  cf.br ^join
^join:
  tvm_ffi.ObjectIncRef %a : !tvm_ffi.c.tensor
  tvm_ffi.ObjectIncRef %b : !tvm_ffi.c.tensor
  return
}

// Unreachable registrations are erased; empty cleanup may split edges.
// CHECK-LABEL: func.func @unreachable_and_empty(
// CHECK-SAME: %[[V:[a-zA-Z0-9_]+]]: !tvm_ffi.c.tensor)
// CHECK-NEXT: cf.br ^[[EDGE:bb[0-9]+]]
// CHECK: llvm.unreachable
// CHECK: ^[[EDGE]]:
// CHECK-NEXT: cf.br ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return
func.func @unreachable_and_empty(%v: !tvm_ffi.c.tensor) {
  tvm_ffi.defer { tvm_ffi.defer_yield }
  cf.br ^exit
^dead:
  tvm_ffi.defer {
    tvm_ffi.ObjectDecRef %v : !tvm_ffi.c.tensor
    tvm_ffi.defer_yield
  }
  llvm.unreachable
^exit:
  return
}
