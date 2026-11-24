# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Execute region reference transfers through packed entry points."""

import unittest

import torch
import tvm_ffi
from trident.core import ir, register_all_passes
from trident.core.execution_engine import ExecutionEngine
from trident.core.passmanager import PassManager
from trident.utils.runtime import find_runtime_libraries

from test.support.cuda import assert_no_cuda_leak
from test.support.ir import new_context


class RegionOwnershipTest(unittest.TestCase):
    def test_loop_carried_references(self) -> None:
        context = new_context()
        with context:
            register_all_passes()
            module = ir.Module.parse("""
                tvm_ffi_api.func @for_loop(%input: !tvm_ffi_api.tensor, %count: !tvm_ffi_abi.int,
                                      %replace: !tvm_ffi_abi.bool) -> !tvm_ffi_api.tensor {
                  %bound = tvm_ffi_api.get %count : !tvm_ffi_abi.int -> i64
                  %condition = tvm_ffi_api.get %replace : !tvm_ffi_abi.bool -> i1
                  %zero = arith.constant 0 : i64
                  %one = arith.constant 1 : i64
                  %result = scf.for %i = %zero to %bound step %one
                      iter_args(%current = %input) -> (!tvm_ffi_api.tensor) : i64 {
                    %next = scf.if %condition -> (!tvm_ffi_api.tensor) {
                      %clone = tvm_ffi_api.tensor.clone %current : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
                      scf.yield %clone : !tvm_ffi_api.tensor
                    } else {
                      scf.yield %current : !tvm_ffi_api.tensor
                    }
                    scf.yield %next : !tvm_ffi_api.tensor
                  }
                  %payload = tvm_ffi_api.upcast %result : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @while_loop(%input: !tvm_ffi_api.tensor, %count: !tvm_ffi_abi.int)
                    -> !tvm_ffi_api.tensor {
                  %bound = tvm_ffi_api.get %count : !tvm_ffi_abi.int -> i64
                  %zero = arith.constant 0 : i64
                  %one = arith.constant 1 : i64
                  %iterations, %result = scf.while (%i = %zero, %current = %input)
                      : (i64, !tvm_ffi_api.tensor) -> (i64, !tvm_ffi_api.tensor) {
                    %condition = arith.cmpi slt, %i, %bound : i64
                    scf.condition(%condition) %i, %current : i64, !tvm_ffi_api.tensor
                  } do {
                  ^body(%i: i64, %current: !tvm_ffi_api.tensor):
                    %next_i = arith.addi %i, %one : i64
                    %next = tvm_ffi_api.tensor.clone %current : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
                    scf.yield %next_i, %next : i64, !tvm_ffi_api.tensor
                  }
                  %payload = tvm_ffi_api.upcast %result : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @discard_loop(%input: !tvm_ffi_api.tensor, %count: !tvm_ffi_abi.int)
                    -> !tvm_ffi_abi.none {
                  %bound = tvm_ffi_api.get %count : !tvm_ffi_abi.int -> i64
                  %zero = arith.constant 0 : i64
                  %one = arith.constant 1 : i64
                  %result = scf.for %i = %zero to %bound step %one
                      iter_args(%current = %input) -> (!tvm_ffi_api.tensor) : i64 {
                    %next = tvm_ffi_api.tensor.clone %current : !tvm_ffi_api.tensor -> !tvm_ffi_api.tensor
                    scf.yield %next : !tvm_ffi_api.tensor
                  }
                  %none = tvm_ffi_abi.constant.none
                  %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
            """)
            PassManager.parse("builtin.module(trident-lowering-pipeline)").run(
                module.operation
            )
            engine = ExecutionEngine(
                module, opt_level=3, shared_libs=find_runtime_libraries()
            )
            engine.initialize()
            for_loop = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_for_loop"), keep_alive_object=engine
            )
            while_loop = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_while_loop"), keep_alive_object=engine
            )
            discard_loop = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_discard_loop"), keep_alive_object=engine
            )

        def check() -> None:
            tensor = torch.arange(16, device="cuda", dtype=torch.float32)
            for count in (0, 1, 5):
                for replace in (False, True):
                    with self.subTest(loop="for", count=count, replace=replace):
                        result = torch.from_dlpack(for_loop(tensor, count, replace))
                        torch.testing.assert_close(result, tensor)
                        self.assertEqual(
                            result.data_ptr() == tensor.data_ptr(),
                            count == 0 or not replace,
                        )
                with self.subTest(loop="while", count=count):
                    result = torch.from_dlpack(while_loop(tensor, count))
                    torch.testing.assert_close(result, tensor)
                    self.assertEqual(result.data_ptr() == tensor.data_ptr(), count == 0)
                with self.subTest(loop="discard", count=count):
                    self.assertIsNone(discard_loop(tensor, count))
            torch.testing.assert_close(
                tensor, torch.arange(16, device="cuda", dtype=torch.float32)
            )

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
