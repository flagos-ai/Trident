// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

// RUN: trident-core-opt %s | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @case_0(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<i1>) -> (i1, i1)
tvm_ffi.func @case_0(%input: !tvm_ffi.monad.result<i1>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<i1>) -> (i1, i1)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_1(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
tvm_ffi.func @case_1(%input: !tvm_ffi.monad.result<i64>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<i64>) -> (i1, i64)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_2(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<f64>) -> (i1, f64)
tvm_ffi.func @case_2(%input: !tvm_ffi.monad.result<f64>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<f64>) -> (i1, f64)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_3(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<index>) -> (i1, index)
tvm_ffi.func @case_3(%input: !tvm_ffi.monad.result<index>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<index>) -> (i1, index)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_4(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<vector<2xi32>>) -> (i1, vector<2xi32>)
tvm_ffi.func @case_4(%input: !tvm_ffi.monad.result<vector<2xi32>>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<vector<2xi32>>) -> (i1, vector<2xi32>)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_5(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
tvm_ffi.func @case_5(%input: !tvm_ffi.monad.result<!tvm_ffi.tensor>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_6(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.array>) -> (i1, !tvm_ffi.any)
tvm_ffi.func @case_6(%input: !tvm_ffi.monad.result<!tvm_ffi.array>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.array>) -> (i1, !tvm_ffi.any)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_7(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.exception>) -> (i1, !tvm_ffi.any)
tvm_ffi.func @case_7(%input: !tvm_ffi.monad.result<!tvm_ffi.exception>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.exception>) -> (i1, !tvm_ffi.any)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_8(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
tvm_ffi.func @case_8(%input: !tvm_ffi.monad.result<!tvm_ffi.any>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.any>) -> (i1, !tvm_ffi.any)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_9(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any)
tvm_ffi.func @case_9(%input: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>>) -> (i1, !tvm_ffi.any)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_10(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.function>) -> (i1, !tvm_ffi.function)
tvm_ffi.func @case_10(%input: !tvm_ffi.monad.result<!tvm_ffi.function>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.function>) -> (i1, !tvm_ffi.function)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_11(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.object>) -> (i1, !tvm_ffi.object)
tvm_ffi.func @case_11(%input: !tvm_ffi.monad.result<!tvm_ffi.object>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.object>) -> (i1, !tvm_ffi.object)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

// CHECK-LABEL: tvm_ffi.func @case_12(
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]], %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.monad.unpack %[[INPUT:[a-zA-Z0-9_]+]] : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>>) -> (i1, !tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>)
tvm_ffi.func @case_12(%input: !tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>>) -> !tvm_ffi.monad.result<!tvm_ffi.none> {
  %success, %value = tvm_ffi.monad.unpack %input : (!tvm_ffi.monad.result<!tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>>) -> (i1, !tvm_ffi.union<!tvm_ffi.function, !tvm_ffi.none>)
  %unit = tvm_ffi.constant.none
  %result = tvm_ffi.monad.pure %unit : (!tvm_ffi.none) -> !tvm_ffi.monad.result<!tvm_ffi.none>
  tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.none>
}

