// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup --verify-each | FileCheck %s
// RUN: trident-core-opt %s --materialize-tvm-ffi-cleanup > %t
// RUN: trident-core-opt %t --materialize-tvm-ffi-cleanup --verify-each | diff %t -

// Retain every exported occurrence before releasing any local, even when the
// export order differs from definition order and the last local is not exported.
// CHECK-LABEL: func.func @exports_before_cleanup(
// CHECK: %[[FIRST:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: %[[SECOND:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: %[[LAST:[a-zA-Z0-9_]+]] = tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FIRST]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LAST]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[SECOND]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FIRST]]
// CHECK-NEXT: return %[[SECOND]], %[[FIRST]], %[[SECOND]]
func.func @exports_before_cleanup() -> (!tvm_ffi.exception, !tvm_ffi.exception, !tvm_ffi.exception) {
  %first = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %second = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  %last = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
  return {tvm_ffi.retain} %second, %first, %second : !tvm_ffi.exception, !tvm_ffi.exception, !tvm_ffi.exception
}

// Skipping a region does not skip its owner's results in the parent block.
// CHECK-LABEL: func.func @region_owner(
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.execute_region -> !tvm_ffi.exception {
// CHECK-NEXT: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup}
// CHECK-NEXT: cf.br ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: scf.yield {tvm_ffi.retain} %[[LOCAL]] : !tvm_ffi.exception
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OUTPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTPUT]]
// CHECK-NEXT: return %[[OUTPUT]]
func.func @region_owner() -> !tvm_ffi.exception {
  %output = scf.execute_region -> !tvm_ffi.exception {
    %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    cf.br ^exit
  ^exit:
    scf.yield {tvm_ffi.retain} %local : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.exception
}

// Selected scalar-only blocks consume their markers without folding constants.
// CHECK-LABEL: func.func @scalar_cleanup(
// CHECK-NEXT: %[[NUMBER:[a-zA-Z0-9_]+]] = arith.constant 1 : i64
// CHECK-NEXT: return %[[NUMBER]] : i64
func.func @scalar_cleanup() -> i64 {
  %number = arith.constant {tvm_ffi.cleanup} 1 : i64
  return {tvm_ffi.retain} %number : i64
}

// Direct markers in a multi-block region remain untouched. The walk driver also
// skips the unreachable block and its nested region.
// The rewrite driver must not fold constants or remove unreachable code.
// CHECK-LABEL: func.func @scalar_markers(
// CHECK: %[[ENTRY:[a-zA-Z0-9_]+]] = arith.constant {tvm_ffi.cleanup} 1 : i64
// CHECK-NEXT: return {tvm_ffi.retain}
// CHECK: %[[UNREACHABLE:[a-zA-Z0-9_]+]] = arith.constant {tvm_ffi.cleanup} 2 : i64
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = scf.execute_region -> i64 {
// CHECK-NEXT: %[[INNER:[a-zA-Z0-9_]+]] = arith.constant {tvm_ffi.cleanup} 3 : i64
// CHECK-NEXT: scf.yield {tvm_ffi.retain} %[[INNER]] : i64
// CHECK-NEXT: } {tvm_ffi.cleanup}
// CHECK-NEXT: return {tvm_ffi.retain}
func.func @scalar_markers() {
  %entry = arith.constant {tvm_ffi.cleanup} 1 : i64
  return {tvm_ffi.retain}
^unreachable:
  %unreachable = arith.constant {tvm_ffi.cleanup} 2 : i64
  %value = scf.execute_region -> i64 {
    %inner = arith.constant {tvm_ffi.cleanup} 3 : i64
    scf.yield {tvm_ffi.retain} %inner : i64
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain}
}

// A previously unsupported single-block region now gets both retain and cleanup.
// Its owning operation's results are cleaned up according to the parent block.
// CHECK-LABEL: func.func @single_block_execute(
// CHECK: %[[OUTPUT:[a-zA-Z0-9_]+]] = scf.execute_region -> !tvm_ffi.exception {
// CHECK-NEXT: %[[LOCAL:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[LOCAL]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[LOCAL]]
// CHECK-NEXT: scf.yield %[[LOCAL]]
// CHECK-NEXT: }
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OUTPUT]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OUTPUT]]
// CHECK-NEXT: return %[[OUTPUT]]
func.func @single_block_execute() -> !tvm_ffi.exception {
  %output = scf.execute_region -> !tvm_ffi.exception {
    %local = tvm_ffi.exception "GuardMatch" {tvm_ffi.cleanup} : !tvm_ffi.exception
    scf.yield {tvm_ffi.retain} %local : !tvm_ffi.exception
  } {tvm_ffi.cleanup}
  return {tvm_ffi.retain} %output : !tvm_ffi.exception
}
