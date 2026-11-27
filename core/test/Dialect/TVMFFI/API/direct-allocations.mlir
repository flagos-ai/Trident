// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
// RUN: trident-core-opt %s --direct-allocations --verify-each | FileCheck %s --check-prefix=MONAD
// RUN: trident-core-opt %s --trident-lowering-pipeline --verify-each | FileCheck %s --check-prefix=LLVM --implicit-check-not='trident.aten.empty' --implicit-check-not='trident.aten.zeros' --implicit-check-not='trident.aten.view' --implicit-check-not='ffi.Array' --implicit-check-not='ffi.StructuralEqual'
// MONAD-LABEL: tvm_ffi_api.func @allocate()
// MONAD: tvm_ffi_monad.direct_allocation 0
// MONAD-NOT: global.call
// MONAD-NOT: function.call
// MONAD-LABEL: tvm_ffi_api.func @allocate_zeros()
// MONAD: tvm_ffi_monad.direct_allocation 3
// MONAD-NOT: global.call
// MONAD-NOT: function.call
// MONAD-LABEL: tvm_ffi_api.func @reshape(
// MONAD: tvm_ffi_monad.direct_allocation 5
// MONAD-NOT: global.call
// MONAD-NOT: function.call
// LLVM-LABEL: llvm.func @__tvm_ffi_allocate(
// LLVM-NOT: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.call @trident_c_allocate_empty(
// LLVM: llvm.icmp "eq"
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_allocate_zeros(
// LLVM-NOT: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.call @trident_c_allocate_zeros(
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_reshape(
// LLVM-NOT: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.call @trident_c_view(
// LLVM: llvm.return
// LLVM-LABEL: llvm.func @__tvm_ffi_scalar_eq(
// LLVM-NOT: llvm.call @TVMFFIFunctionCall
// LLVM: llvm.icmp "eq"
// LLVM: llvm.return
module {
 tvm_ffi_api.func @allocate() -> !tvm_ffi_api.tensor {
  %n = tvm_ffi_abi.constant.int 8
  %sizes = tvm_ffi_api.array.create(%n, %n) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  %none = tvm_ffi_abi.constant.none
  %tensor = tvm_ffi_api.global.call "trident.aten.empty.memory_format"(%sizes, %none, %none, %none, %none, %none) : (!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.none, !tvm_ffi_abi.none, !tvm_ffi_abi.none, !tvm_ffi_abi.none, !tvm_ffi_abi.none) -> !tvm_ffi_api.tensor
  %result = tvm_ffi_monad.pure %tensor : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  %bound = tvm_ffi_monad.bind %result : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor> {
  ^bb0(%t: !tvm_ffi_api.tensor):
    %p = tvm_ffi_monad.pure %t : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
    tvm_ffi_monad.yield %p : !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  }
  %success, %value = tvm_ffi_monad.unpack %bound : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (i1, !tvm_ffi_api.any)
  %flag = tvm_ffi_abi.to %success : i1 -> !tvm_ffi_abi.bool
  tvm_ffi_api.return %flag, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
 }

 tvm_ffi_api.func @allocate_zeros() -> !tvm_ffi_api.tensor {
  %n = tvm_ffi_abi.constant.int 8
  %sizes = tvm_ffi_api.array.create(%n, %n) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  %none = tvm_ffi_abi.constant.none
  %tensor = tvm_ffi_api.global.call "trident.aten.zeros"(%sizes, %none, %none, %none, %none) : (!tvm_ffi_api.array<!tvm_ffi_abi.int>, !tvm_ffi_abi.none, !tvm_ffi_abi.none, !tvm_ffi_abi.none, !tvm_ffi_abi.none) -> !tvm_ffi_api.tensor
  %result = tvm_ffi_monad.pure %tensor : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  %success, %value = tvm_ffi_monad.unpack %result : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (i1, !tvm_ffi_api.any)
  %flag = tvm_ffi_abi.to %success : i1 -> !tvm_ffi_abi.bool
  tvm_ffi_api.return %flag, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
 }

 tvm_ffi_api.func @reshape(%x: !tvm_ffi_api.tensor) -> !tvm_ffi_api.tensor {
  %n = tvm_ffi_abi.constant.int 8
  %sizes = tvm_ffi_api.array.create(%n, %n) : (!tvm_ffi_abi.int, !tvm_ffi_abi.int) -> !tvm_ffi_api.array<!tvm_ffi_abi.int>
  %handle = tvm_ffi_api.function.get_global "trident.aten.view" : !tvm_ffi_api.function
  %tensor = tvm_ffi_api.function.call %handle : !tvm_ffi_api.function(%x, %sizes) : (!tvm_ffi_api.tensor, !tvm_ffi_api.array<!tvm_ffi_abi.int>) -> !tvm_ffi_api.tensor
  %result = tvm_ffi_monad.pure %tensor : (!tvm_ffi_api.tensor) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
  %success, %value = tvm_ffi_monad.unpack %result : (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (i1, !tvm_ffi_api.any)
  %flag = tvm_ffi_abi.to %success : i1 -> !tvm_ffi_abi.bool
  tvm_ffi_api.return %flag, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
 }

 tvm_ffi_api.func @scalar_eq(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int) -> !tvm_ffi_abi.bool {
  %eq = tvm_ffi_abi.eq %a, %b : !tvm_ffi_abi.int, !tvm_ffi_abi.int
  %boxed = tvm_ffi_abi.to %eq : i1 -> !tvm_ffi_abi.bool
  %ok = tvm_ffi_abi.constant.bool true
  %any = tvm_ffi_api.upcast %boxed : !tvm_ffi_abi.bool -> !tvm_ffi_api.any
  tvm_ffi_api.return %ok, %any : !tvm_ffi_abi.bool, !tvm_ffi_api.any
 }
}
