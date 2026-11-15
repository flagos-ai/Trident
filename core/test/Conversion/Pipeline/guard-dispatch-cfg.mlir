// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --implicit-check-not='!tvm_ffi.' --implicit-check-not='unrealized_conversion_cast'

// Reduced from a ten-version shared-prefix dispatcher. Repeated metadata reads
// remain effectful, and sibling guards share their fallback blocks. Structuring
// and lowering this CFG introduces forwarding joins. Explicit canonicalization
// must simplify them before GPU vector preparation runs.
// Preserve block order: the fallback chain is deliberately emitted before its
// predecessors, as in the guard tree's reverse construction.
// CHECK-LABEL: llvm.func @__tvm_ffi_dispatch_shared_fallbacks(
// CHECK-SAME: %[[ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[COUNT:[a-zA-Z0-9_]+]]: i32,
// CHECK: %[[STATUS0:[a-zA-Z0-9_]+]] = llvm.call @__tvm_ffi_candidate_0(%[[CTX:[a-zA-Z0-9_]+]], %[[ARGS]], %[[COUNT]], %[[OUTPUT0:[a-zA-Z0-9_]+]])
// CHECK: %[[OK0:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS0]], %[[ZERO:[a-zA-Z0-9_]+]] : i32
// CHECK: llvm.cond_br %[[OK0]]
// CHECK: llvm.load %[[OUTPUT0]]
// CHECK: %[[STATUS9:[a-zA-Z0-9_]+]] = llvm.call @__tvm_ffi_candidate_9(%[[CTX]], %[[ARGS]], %[[COUNT]], %[[OUTPUT9:[a-zA-Z0-9_]+]])
// CHECK: %[[OK9:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[STATUS9]], %[[ZERO]] : i32
// CHECK: llvm.cond_br %[[OK9]]
// CHECK: llvm.load %[[OUTPUT9]]
// CHECK: llvm.return

module {
  tvm_ffi.func private @candidate_0(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_1(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_2(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_3(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_4(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_5(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_6(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_7(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_8(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.func private @candidate_9(!tvm_ffi.tensor, !tvm_ffi.tensor) -> !tvm_ffi.any
  tvm_ffi.pack.func @dispatch_shared_fallbacks(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c32_i64 = arith.constant 32 : i64
    %c4097_i64 = arith.constant 4097 : i64
    %c4098_i64 = arith.constant 4098 : i64
    %c4099_i64 = arith.constant 4099 : i64
    %c4100_i64 = arith.constant 4100 : i64
    %c4101_i64 = arith.constant 4101 : i64
    %c4102_i64 = arith.constant 4102 : i64
    %c4103_i64 = arith.constant 4103 : i64
    %c4104_i64 = arith.constant 4104 : i64
    %c4105_i64 = arith.constant 4105 : i64
    %0:2 = scf.execute_region -> (!tvm_ffi.bool, !tvm_ffi.any) no_inline {
      %1 = tvm_ffi.pack.get %args[%c0_i64] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.any
      %2 = tvm_ffi.downcast %1 : !tvm_ffi.any -> !tvm_ffi.tensor
      %3 = tvm_ffi.pack.get %args[%c1_i64] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.any
      %4 = tvm_ffi.downcast %3 : !tvm_ffi.any -> !tvm_ffi.tensor
      %code, %bits, %lanes = tvm_ffi.tensor.dtype %4 : !tvm_ffi.tensor
      %5 = arith.extui %code : i8 to i64
      %6 = arith.extui %bits : i8 to i64
      %7 = arith.extui %lanes : i16 to i64
      %8 = arith.cmpi eq, %5, %c2_i64 : i64
      %9 = arith.cmpi eq, %6, %c32_i64 : i64
      %10 = arith.cmpi eq, %7, %c1_i64 : i64
      %11 = arith.andi %8, %9 : i1
      %12 = arith.andi %11, %10 : i1
      cf.cond_br %12, ^bb2, ^bb1
    ^bb1:
      %13 = tvm_ffi.exception "GuardMiss" : !tvm_ffi.exception
      %14 = tvm_ffi.upcast %13 : !tvm_ffi.exception -> !tvm_ffi.any
      %15 = tvm_ffi.constant.bool true
      scf.yield %15, %14 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb2:
      %16 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %17 = arith.cmpi eq, %16, %c1_i64 : i64
      cf.cond_br %17, ^bb29, ^bb26
    ^bb3:
      %18 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      cf.cond_br %19, ^bb4, ^bb1
    ^bb4:
      %success, %value = tvm_ffi.pack.call @candidate_9(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success, %value : !tvm_ffi.bool, !tvm_ffi.any
    ^bb5:
      %20 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %21 = arith.cmpi eq, %20, %c1_i64 : i64
      cf.cond_br %21, ^bb6, ^bb3
    ^bb6:
      %22 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %23 = arith.cmpi eq, %22, %c4105_i64 : i64
      cf.cond_br %23, ^bb7, ^bb3
    ^bb7:
      %success_0, %value_1 = tvm_ffi.pack.call @candidate_8(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_0, %value_1 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb8:
      %24 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      cf.cond_br %25, ^bb9, ^bb5
    ^bb9:
      %26 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %27 = arith.cmpi eq, %26, %c4104_i64 : i64
      cf.cond_br %27, ^bb10, ^bb5
    ^bb10:
      %success_2, %value_3 = tvm_ffi.pack.call @candidate_7(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_2, %value_3 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb11:
      %28 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %29 = arith.cmpi eq, %28, %c1_i64 : i64
      cf.cond_br %29, ^bb12, ^bb8
    ^bb12:
      %30 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %31 = arith.cmpi eq, %30, %c4103_i64 : i64
      cf.cond_br %31, ^bb13, ^bb8
    ^bb13:
      %success_4, %value_5 = tvm_ffi.pack.call @candidate_6(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_4, %value_5 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb14:
      %32 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %33 = arith.cmpi eq, %32, %c1_i64 : i64
      cf.cond_br %33, ^bb15, ^bb11
    ^bb15:
      %34 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %35 = arith.cmpi eq, %34, %c4102_i64 : i64
      cf.cond_br %35, ^bb16, ^bb11
    ^bb16:
      %success_6, %value_7 = tvm_ffi.pack.call @candidate_5(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_6, %value_7 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb17:
      %36 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %37 = arith.cmpi eq, %36, %c1_i64 : i64
      cf.cond_br %37, ^bb18, ^bb14
    ^bb18:
      %38 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %39 = arith.cmpi eq, %38, %c4101_i64 : i64
      cf.cond_br %39, ^bb19, ^bb14
    ^bb19:
      %success_8, %value_9 = tvm_ffi.pack.call @candidate_4(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_8, %value_9 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb20:
      %40 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %41 = arith.cmpi eq, %40, %c1_i64 : i64
      cf.cond_br %41, ^bb21, ^bb17
    ^bb21:
      %42 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %43 = arith.cmpi eq, %42, %c4100_i64 : i64
      cf.cond_br %43, ^bb22, ^bb17
    ^bb22:
      %success_10, %value_11 = tvm_ffi.pack.call @candidate_3(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_10, %value_11 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb23:
      %44 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %45 = arith.cmpi eq, %44, %c1_i64 : i64
      cf.cond_br %45, ^bb24, ^bb20
    ^bb24:
      %46 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %47 = arith.cmpi eq, %46, %c4099_i64 : i64
      cf.cond_br %47, ^bb25, ^bb20
    ^bb25:
      %success_12, %value_13 = tvm_ffi.pack.call @candidate_2(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_12, %value_13 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb26:
      %48 = tvm_ffi.tensor.stride %2[%c0_i64] : !tvm_ffi.tensor
      %49 = arith.cmpi eq, %48, %c1_i64 : i64
      cf.cond_br %49, ^bb27, ^bb23
    ^bb27:
      %50 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %51 = arith.cmpi eq, %50, %c4098_i64 : i64
      cf.cond_br %51, ^bb28, ^bb23
    ^bb28:
      %success_14, %value_15 = tvm_ffi.pack.call @candidate_1(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_14, %value_15 : !tvm_ffi.bool, !tvm_ffi.any
    ^bb29:
      %52 = tvm_ffi.tensor.size %4[%c0_i64] : !tvm_ffi.tensor
      %53 = arith.cmpi eq, %52, %c4097_i64 : i64
      cf.cond_br %53, ^bb30, ^bb26
    ^bb30:
      %success_16, %value_17 = tvm_ffi.pack.call @candidate_0(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
      scf.yield %success_16, %value_17 : !tvm_ffi.bool, !tvm_ffi.any
    }
    tvm_ffi.return %0#0, %0#1 : !tvm_ffi.bool, !tvm_ffi.any
  }
}

