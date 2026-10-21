# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Execute local packed calls after monad and container lowering."""

import unittest

import torch
import tvm_ffi
from trident import capi_utils
from trident.core import ir, register_all_passes
from trident.core.execution_engine import ExecutionEngine
from trident.core.passmanager import PassManager

from test.support.cuda import assert_no_cuda_leak
from test.support.ir import new_context


class LocalCallTest(unittest.TestCase):
    def test_erased_packed_boundaries(self) -> None:
        context = new_context()
        with context:
            register_all_passes()
            module = ir.Module.parse("""
                tvm_ffi.func @finite(%value: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>)
                    -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>
                    attributes {emit_tvm_ffi_abi} {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool true
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @string(%value: !tvm_ffi.str) -> !tvm_ffi.str
                    attributes {emit_tvm_ffi_abi} {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.str -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool true
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @failed() -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float>
                    attributes {emit_tvm_ffi_abi} {
                  %none = tvm_ffi.constant.none
                  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool false
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_failed() -> !tvm_ffi.bool
                    attributes {emit_tvm_ffi_abi} {
                  %ok, %value = tvm_ffi.call @failed() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %payload = tvm_ffi.upcast %ok : !tvm_ffi.bool -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
            PassManager.parse("builtin.module(trident-lowering-pipeline)").run(
                module.operation
            )
            engine = ExecutionEngine(
                module, opt_level=3, shared_libs=capi_utils.find_runtime_libraries()
            )
            engine.initialize()
            finite = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_finite"), keep_alive_object=engine
            )
            string = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_string"), keep_alive_object=engine
            )
            failed = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_invoke_failed"), keep_alive_object=engine
            )

        def check() -> None:
            for value in (None, -7, 42):
                self.assertEqual(finite(value), value)
            for value in ("", "1234567", "12345678", "a longer heap string"):
                self.assertEqual(string(value), value)
            # Union/Str constraints are erased to CAny during decomposition.
            for value in (1.5, "wrong tag", True):
                self.assertEqual(finite(value), value)
            self.assertEqual(string(7), 7)
            # Failure must skip output-slot access, even when the payload is
            # outside the declared semantic result set.
            self.assertFalse(failed())

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_packed_call_results(self) -> None:
        context = new_context()
        with context:
            register_all_passes()
            module = ir.Module.parse("""
                tvm_ffi.func @callee(%ok: !tvm_ffi.bool, %value: !tvm_ffi.tensor,
                                     %unused: !tvm_ffi.int) -> !tvm_ffi.tensor
                    attributes {emit_tvm_ffi_abi} {
                  %success = tvm_ffi.get %ok : !tvm_ffi.bool -> i1
                  %payload = scf.if %success -> (!tvm_ffi.any) {
                    %value_any = tvm_ffi.upcast %value : !tvm_ffi.tensor -> !tvm_ffi.any
                    scf.yield %value_any : !tvm_ffi.any
                  } else {
                    %none = tvm_ffi.constant.none
                    %none_any = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                    scf.yield %none_any : !tvm_ffi.any
                  }
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke(%ok: !tvm_ffi.bool, %value: !tvm_ffi.tensor,
                                     %unused: !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                    attributes {emit_tvm_ffi_abi} {
                  %call_ok, %call_value = tvm_ffi.call @callee(%ok, %value, %unused) :
                    (!tvm_ffi.bool, !tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %result = tvm_ffi.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                  %payload = tvm_ffi.upcast %result : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_monad(%ok: !tvm_ffi.bool, %value: !tvm_ffi.tensor,
                                           %unused: !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                    attributes {emit_tvm_ffi_abi} {
                  %call = tvm_ffi.monad.call @callee(%ok, %value, %unused) :
                    (!tvm_ffi.bool, !tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.tensor>
                  %call_ok, %call_value = tvm_ffi.monad.unpack %call :
                    (!tvm_ffi.monad.result<!tvm_ffi.tensor>) -> (i1, !tvm_ffi.any)
                  %flag = tvm_ffi.to %call_ok : i1 -> !tvm_ffi.bool
                  %result = tvm_ffi.tuple.construct(%flag, %call_value) :
                    (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                  %payload = tvm_ffi.upcast %result : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @empty() -> !tvm_ffi.none attributes {emit_tvm_ffi_abi} {
                  %none = tvm_ffi.constant.none
                  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_empty() -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                    attributes {emit_tvm_ffi_abi} {
                  %call_ok, %call_value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %result = tvm_ffi.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                  %payload = tvm_ffi.upcast %result : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
            PassManager.parse("builtin.module(trident-lowering-pipeline)").run(
                module.operation
            )
            engine = ExecutionEngine(
                module, opt_level=3, shared_libs=capi_utils.find_runtime_libraries()
            )
            engine.initialize()
            invoke = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_invoke"), keep_alive_object=engine
            )
            invoke_monad = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_invoke_monad"), keep_alive_object=engine
            )
            invoke_empty = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_invoke_empty"), keep_alive_object=engine
            )

        def check() -> None:
            tensor = torch.arange(4, device="cuda", dtype=torch.float32)
            for name, function in (("direct", invoke), ("monadic", invoke_monad)):
                for ok in (True, False, True):
                    with self.subTest(path=name, success=ok):
                        result = function(ok, tensor, 7)
                        self.assertEqual(result[0], ok)
                        if ok:
                            payload = torch.from_dlpack(result[1])
                            torch.testing.assert_close(payload, tensor)
                            self.assertEqual(payload.data_ptr(), tensor.data_ptr())
                        else:
                            self.assertIsNone(result[1])
            empty = invoke_empty()
            self.assertTrue(empty[0])
            self.assertIsNone(empty[1])

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
