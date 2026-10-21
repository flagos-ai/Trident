// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --lower-tvm-ffi-monad --canonicalize --verify-each | FileCheck %s
// RUN: trident-core-opt %s --lower-tvm-ffi-monad --convert-scf-to-cf --canonicalize --verify-each | FileCheck %s --check-prefix=CFG
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM

// CFG-LABEL:   tvm_ffi.func @checked_continuation(
// CFG-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi} {
// CFG:           %[[CALL_0:[a-zA-Z0-9_$.-]+]]:2 = func.call @lookup() : () -> (!tvm_ffi.c.function, i1)
// CFG:           cf.cond_br %[[CALL_0]]#1, ^[[BLOCK_bb1:bb[0-9]+]], ^[[BLOCK_bb2:bb[0-9]+]]
// CFG-DAG:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_0]]#0 : !tvm_ffi.c.function -> !tvm_ffi.any
// CFG-DAG:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CFG:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb3:bb[0-9]+]](%[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_1]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb3]](%[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:         ^[[BLOCK_bb3]](%[[VAL_0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CFG:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[VAL_0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CFG:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CFG:           cf.cond_br %[[GET_0]], ^[[BLOCK_bb4:bb[0-9]+]], ^[[BLOCK_bb12:bb[0-9]+]]
// CFG:           %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.c.function
// CFG:           %[[CALL_1:[a-zA-Z0-9_$.-]+]]:2 = func.call @checked_call(%[[DOWNCAST_0]], %[[ARG0]]) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
// CFG:           cf.cond_br %[[CALL_1]]#1, ^[[BLOCK_bb5:bb[0-9]+]], ^[[BLOCK_bb6:bb[0-9]+]]
// CFG-DAG:           %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_1]]#0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
// CFG-DAG:           %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CFG:           %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb7:bb[0-9]+]](%[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb7]](%[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:         ^[[BLOCK_bb7]](%[[VAL_1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CFG:           %[[TUPLE_5:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[VAL_1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CFG:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_5]]#0 : !tvm_ffi.bool -> i1
// CFG:           cf.cond_br %[[GET_1]], ^[[BLOCK_bb8:bb[0-9]+]], ^[[BLOCK_bb11:bb[0-9]+]]
// CFG:           %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_5]]#1 : !tvm_ffi.any -> !tvm_ffi.c.tensor
// CFG:           %[[CALL_2:[a-zA-Z0-9_$.-]+]]:2 = func.call @checked_call(%[[DOWNCAST_0]], %[[ARG0]]) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
// CFG:           cf.cond_br %[[CALL_2]]#1, ^[[BLOCK_bb9:bb[0-9]+]], ^[[BLOCK_bb10:bb[0-9]+]]
// CFG-DAG:           %[[UPCAST_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_2]]#0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
// CFG-DAG:           %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CFG:           %[[TUPLE_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_4]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb13:bb[0-9]+]](%[[TUPLE_6]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_7]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_8]], %[[UPCAST_5]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb13]](%[[TUPLE_7]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_9]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_10]], %[[UPCAST_6]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb13]](%[[TUPLE_8]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_11]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_12]], %[[UPCAST_7]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb13]](%[[TUPLE_9]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:         ^[[BLOCK_bb13]](%[[VAL_2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CFG:           %[[TUPLE_10:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[VAL_2]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CFG:           %[[GET_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_10]]#0 : !tvm_ffi.bool -> i1
// CFG:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_2]] : i1 -> !tvm_ffi.bool
// CFG:           tvm_ffi.return %[[TO_0]], %[[TUPLE_10]]#1 : !tvm_ffi.bool, !tvm_ffi.any
// CFG-LABEL:   tvm_ffi.func @none_continuation(
// CFG-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CFG-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor,
// CFG-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.none {
// CFG:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           cf.cond_br %[[ARG0]], ^[[BLOCK_bb1:bb[0-9]+]], ^[[BLOCK_bb2:bb[0-9]+]]
// CFG-DAG:           %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG-DAG:           %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CFG:           %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb3:bb[0-9]+]](%[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_2]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb3]](%[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:         ^[[BLOCK_bb3]](%[[VAL_0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CFG:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[VAL_0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CFG:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CFG:           cf.cond_br %[[GET_0]], ^[[BLOCK_bb4:bb[0-9]+]], ^[[BLOCK_bb5:bb[0-9]+]]
// CFG:           %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.none
// CFG:           tvm_ffi.tensor.copy_ %[[ARG2]], %[[ARG1]] : !tvm_ffi.c.tensor
// CFG-DAG:           %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG-DAG:           %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CFG:           %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb6:bb[0-9]+]](%[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:           %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CFG:           %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CFG:           %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CFG:           %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CFG:           cf.br ^[[BLOCK_bb6]](%[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>)
// CFG:         ^[[BLOCK_bb6]](%[[VAL_1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>):
// CFG:           %[[TUPLE_5:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[VAL_1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CFG:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_5]]#0 : !tvm_ffi.bool -> i1
// CFG:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_1]] : i1 -> !tvm_ffi.bool
// CFG:           tvm_ffi.return %[[TO_0]], %[[TUPLE_5]]#1 : !tvm_ffi.bool, !tvm_ffi.any


// CHECK-LABEL:   tvm_ffi.func @checked_continuation(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi} {
// CHECK:           %[[CALL_0:[a-zA-Z0-9_$.-]+]]:2 = func.call @lookup() : () -> (!tvm_ffi.c.function, i1)
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[CALL_0]]#1 -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_0]]#0 : !tvm_ffi.c.function -> !tvm_ffi.any
// CHECK-DAG:             %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_0]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_1]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_2]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.c.function
// CHECK:             %[[CALL_1:[a-zA-Z0-9_$.-]+]]:2 = func.call @checked_call(%[[DOWNCAST_0]], %[[ARG0]]) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
// CHECK:             %[[IF_2:[a-zA-Z0-9_$.-]+]] = scf.if %[[CALL_1]]#1 -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG:               %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_1]]#0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
// CHECK-DAG:               %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:               %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               scf.yield %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             } else {
// CHECK:               %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:               %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:               %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_4]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:               %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_5]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               scf.yield %[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             %[[TUPLE_5:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_2]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:             %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_5]]#0 : !tvm_ffi.bool -> i1
// CHECK:             %[[IF_3:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_1]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:               %[[DOWNCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_5]]#1 : !tvm_ffi.any -> !tvm_ffi.c.tensor
// CHECK:               %[[CALL_2:[a-zA-Z0-9_$.-]+]]:2 = func.call @checked_call(%[[DOWNCAST_0]], %[[ARG0]]) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
// CHECK:               %[[IF_4:[a-zA-Z0-9_$.-]+]] = scf.if %[[CALL_2]]#1 -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG:                 %[[UPCAST_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CALL_2]]#0 : !tvm_ffi.c.tensor -> !tvm_ffi.any
// CHECK-DAG:                 %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:                 %[[TUPLE_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_4]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:                 scf.yield %[[TUPLE_6]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               } else {
// CHECK:                 %[[CONSTANT_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:                 %[[CONSTANT_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:                 %[[UPCAST_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_7]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:                 %[[TUPLE_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_8]], %[[UPCAST_5]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:                 scf.yield %[[TUPLE_7]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               scf.yield %[[IF_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             } else {
// CHECK:               %[[CONSTANT_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:               %[[CONSTANT_10:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:               %[[UPCAST_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_9]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:               %[[TUPLE_8:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_10]], %[[UPCAST_6]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:               scf.yield %[[TUPLE_8]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[IF_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_11:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_12:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_7:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_11]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_9:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_12]], %[[UPCAST_7]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_9]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_10:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_10]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_2]] : i1 -> !tvm_ffi.bool
// CHECK:           tvm_ffi.return %[[TO_0]], %[[TUPLE_10]]#1 : !tvm_ffi.bool, !tvm_ffi.any
// CHECK-LABEL:   tvm_ffi.func @none_continuation(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9_$.-]+]]: i1,
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor,
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9_$.-]+]]: !tvm_ffi.c.tensor) -> !tvm_ffi.none {
// CHECK:           %[[CONSTANT_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:           %[[IF_0:[a-zA-Z0-9_$.-]+]] = scf.if %[[ARG0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK-DAG:             %[[UPCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK-DAG:             %[[CONSTANT_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_1]], %[[UPCAST_0]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_0]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_2]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_3]], %[[UPCAST_1]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_1]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_2:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_0]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_2]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[IF_1:[a-zA-Z0-9_$.-]+]] = scf.if %[[GET_0]] -> (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) {
// CHECK:             %[[DOWNCAST_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.downcast %[[TUPLE_2]]#1 : !tvm_ffi.any -> !tvm_ffi.none
// CHECK:             tvm_ffi.tensor.copy_ %[[ARG2]], %[[ARG1]] : !tvm_ffi.c.tensor
// CHECK-DAG:             %[[UPCAST_2:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[DOWNCAST_0]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK-DAG:             %[[CONSTANT_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool true
// CHECK:             %[[TUPLE_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_4]], %[[UPCAST_2]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_3]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           } else {
// CHECK:             %[[CONSTANT_5:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.none
// CHECK:             %[[CONSTANT_6:[a-zA-Z0-9_$.-]+]] = tvm_ffi.constant.bool false
// CHECK:             %[[UPCAST_3:[a-zA-Z0-9_$.-]+]] = tvm_ffi.upcast %[[CONSTANT_5]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK:             %[[TUPLE_4:[a-zA-Z0-9_$.-]+]] = tvm_ffi.tuple.construct(%[[CONSTANT_6]], %[[UPCAST_3]]) : (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:             scf.yield %[[TUPLE_4]] : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
// CHECK:           %[[TUPLE_5:[a-zA-Z0-9_$.-]+]]:2 = tvm_ffi.tuple.unconstruct %[[IF_1]] : (!tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>) -> (!tvm_ffi.bool, !tvm_ffi.any)
// CHECK:           %[[GET_1:[a-zA-Z0-9_$.-]+]] = tvm_ffi.get %[[TUPLE_5]]#0 : !tvm_ffi.bool -> i1
// CHECK:           %[[TO_0:[a-zA-Z0-9_$.-]+]] = tvm_ffi.to %[[GET_1]] : i1 -> !tvm_ffi.bool
// CHECK:           tvm_ffi.return %[[TO_0]], %[[TUPLE_5]]#1 : !tvm_ffi.bool, !tvm_ffi.any

// LLVM-LABEL: llvm.func @checked_continuation(
// LLVM: llvm.call @checked_call
// LLVM: llvm.cond_br
// LLVM: llvm.call @checked_call
// LLVM: llvm.cond_br
// LLVM: llvm.store
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_checked_continuation(
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @none_continuation(
// LLVM: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.return

// Calls transfer their owned payloads into Results. Explicit cleanup releases
// each Result after the receiving scope acquires its own export reference.
tvm_ffi.func @checked_continuation(%tensor: !tvm_ffi.c.tensor) -> !tvm_ffi.c.tensor attributes {emit_tvm_ffi_abi} {
  %handle, %lookup = func.call @lookup() : () -> (!tvm_ffi.c.function, i1)
  %ready = scf.if %lookup -> (!tvm_ffi.monad.result<!tvm_ffi.c.function>) {
    %pure = tvm_ffi.monad.pure %handle : (!tvm_ffi.c.function) -> !tvm_ffi.monad.result<!tvm_ffi.c.function>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.function>
  } else {
    %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.function>
    scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.function>
  }
  %output = tvm_ffi.monad.bind %ready : (!tvm_ffi.monad.result<!tvm_ffi.c.function>) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
  ^bb0(%callee: !tvm_ffi.c.function):
    %value, %status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
    %first = scf.if %status -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
      %pure = tvm_ffi.monad.pure %value : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    } else {
      %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    }
    %continued = tvm_ffi.monad.bind %first : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor> {
    ^bb0(%unused: !tvm_ffi.c.tensor):
      // Deliberately independent of the first payload: effects still short-circuit.
      %second, %second_status = func.call @checked_call(%callee, %tensor) : (!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
      %last = scf.if %second_status -> (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) {
        %pure = tvm_ffi.monad.pure %second : (!tvm_ffi.c.tensor) -> !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
        scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      } else {
        %failed = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
        scf.yield %failed : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
      }
      tvm_ffi.monad.yield %last : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
    }
    tvm_ffi.monad.yield %continued : !tvm_ffi.monad.result<!tvm_ffi.c.tensor>
  }
  %return_ok_1, %return_value_1 = tvm_ffi.monad.unpack %output : (!tvm_ffi.monad.result<!tvm_ffi.c.tensor>) -> (i1, !tvm_ffi.any)
  %return_flag_1 = tvm_ffi.to %return_ok_1 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_1, %return_value_1 : !tvm_ffi.bool, !tvm_ffi.any
}

// A None payload preserves the status dependency for independent side effects.
tvm_ffi.func @none_continuation(%ready: i1, %source: !tvm_ffi.c.tensor, %destination: !tvm_ffi.c.tensor) -> !tvm_ffi.none {
  %none = tvm_ffi.constant.none
  %input = scf.if %ready -> (!tvm_ffi.monad.result<!tvm_ffi.none>) {
    %pure = tvm_ffi.monad.pure %none : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  } else {
    %fail = tvm_ffi.monad.fail : !tvm_ffi.monad.result<!tvm_ffi.none>
    scf.yield %fail : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %output = tvm_ffi.monad.bind %input : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  ^bb0(%payload: !tvm_ffi.none):
    tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.c.tensor
    %pure = tvm_ffi.monad.pure %payload : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
    tvm_ffi.monad.yield %pure : !tvm_ffi.monad.result<!tvm_ffi.none>
  }
  %return_ok_2, %return_value_2 = tvm_ffi.monad.unpack %output : (!tvm_ffi.monad.result<!tvm_ffi.none>) -> (i1, !tvm_ffi.any)
  %return_flag_2 = tvm_ffi.to %return_ok_2 : i1 -> !tvm_ffi.bool
  tvm_ffi.return %return_flag_2, %return_value_2 : !tvm_ffi.bool, !tvm_ffi.any
}

func.func private @lookup() -> (!tvm_ffi.c.function, i1)
func.func private @checked_call(!tvm_ffi.c.function, !tvm_ffi.c.tensor) -> (!tvm_ffi.c.tensor, i1)
