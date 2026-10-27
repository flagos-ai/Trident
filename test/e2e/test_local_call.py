# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Execute local packed calls after monad and container lowering."""

import unittest
from unittest.mock import patch

import torch
import tvm_ffi
from trident import capi_utils
from trident.backend import TridentGraphModule
from trident.core import ir, register_all_passes
from trident.core.execution_engine import ExecutionEngine
from trident.core.passmanager import PassManager
from trident.ffi import Exception

from test.support.cuda import assert_no_cuda_leak
from test.support.ir import new_context


class LocalCallTest(unittest.TestCase):
    @staticmethod
    def _compile_entries(
        module: ir.Module, *names: str
    ) -> tuple[tvm_ffi.Function, ...]:
        PassManager.parse("builtin.module(trident-lowering-pipeline)").run(
            module.operation
        )
        engine = ExecutionEngine(
            module, opt_level=3, shared_libs=capi_utils.find_runtime_libraries()
        )
        engine.initialize()
        return tuple(
            tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup(f"__tvm_ffi_{name}"), keep_alive_object=engine
            )
            for name in names
        )

    def test_dispatcher_outcomes(self) -> None:
        def dispatch(value):
            return value

        graph = TridentGraphModule(dispatch)
        # Execute the dispatcher directly so its sentinel does not trigger export.
        empty = graph.executor
        graph._output_spec = torch.utils._pytree.tree_structure(0)
        with graph.ctx:
            module = ir.Module.parse("""
                tvm_ffi.func @dispatch_0(%value: !tvm_ffi.tensor) -> !tvm_ffi.any {
                  %rejected = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
                  %payload = tvm_ffi.upcast %rejected : !tvm_ffi.exception -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @dispatch_1(%value: !tvm_ffi.int) -> !tvm_ffi.any {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @dispatch_2() -> !tvm_ffi.int {
                  %answer = tvm_ffi.constant.int 42
                  %payload = tvm_ffi.upcast %answer : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @dispatch_3(%first: !tvm_ffi.int, %second: !tvm_ffi.int) -> !tvm_ffi.int {
                  %payload = tvm_ffi.upcast %first : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
        graph._sub_modules = [
            module,
            *(ir.Module.create(loc=module.operation.location) for _ in range(3)),
        ]
        # Bypass only Python signature normalization to exercise MLIR dynamic
        # arity. This does not claim Python *args/**kwargs guard mapping support.
        with patch(
            "tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature",
            side_effect=lambda invoke, signature: invoke,
        ):
            execute = graph.stub_compile()

        with graph.ctx:
            failure = ir.Module.parse(r"""
                llvm.mlir.global internal constant @error_kind("RuntimeError\00")
                llvm.mlir.global internal constant @error_message("dispatcher failure\00")
                llvm.func @TVMFFIErrorSetRaisedFromCStr(!llvm.ptr, !llvm.ptr)
                tvm_ffi.func @dispatch_0(%value: !tvm_ffi.any) -> !tvm_ffi.any {
                  %kind = llvm.mlir.addressof @error_kind : !llvm.ptr
                  %message = llvm.mlir.addressof @error_message : !llvm.ptr
                  llvm.call @TVMFFIErrorSetRaisedFromCStr(%kind, %message) : (!llvm.ptr, !llvm.ptr) -> ()
                  %none = tvm_ffi.constant.none
                  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                  %no = tvm_ffi.constant.bool false
                  tvm_ffi.return %no, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
            success = module.body.operations[1].operation.clone()
            failure.body.append(success)
        graph._sub_modules = [failure, ir.Module.create(loc=module.operation.location)]
        failed = graph.stub_compile()

        def check() -> None:
            self.assertIsInstance(empty(7), Exception)
            self.assertEqual(execute(), 42)
            self.assertEqual(execute(7, 8), 7)
            self.assertIsInstance(execute(7, 8, 9), Exception)
            self.assertIsInstance(execute(7, "wrong type"), Exception)
            self.assertEqual(execute(7), 7)
            self.assertIsInstance(execute("all rejected"), Exception)
            tensor = torch.ones(2, device="cuda")
            self.assertIsInstance(execute(tensor), Exception)
            self.assertEqual(execute(-3), -3)
            with self.assertRaisesRegex(RuntimeError, "dispatcher failure"):
                failed(7)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_erased_packed_boundaries(self) -> None:
        context = new_context()
        with context:
            register_all_passes()
            module = ir.Module.parse("""
                tvm_ffi.func @finite(%value: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none>)
                    -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.none> -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool true
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @string(%value: !tvm_ffi.str) -> !tvm_ffi.str {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.str -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool true
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @failed() -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.float> {
                  %none = tvm_ffi.constant.none
                  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                  %ok = tvm_ffi.constant.bool false
                  tvm_ffi.return %ok, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_failed() -> !tvm_ffi.bool {
                  %ok, %value = tvm_ffi.call @failed() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %payload = tvm_ffi.upcast %ok : !tvm_ffi.bool -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
            finite, string, failed = self._compile_entries(
                module, "finite", "string", "invoke_failed"
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
                tvm_ffi.pack.func @count(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.int {
                  %count = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
                  %boxed = tvm_ffi.to %count : i64 -> !tvm_ffi.int
                  %value = tvm_ffi.upcast %boxed : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %value : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.pack.func @forward_empty(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
                  %ok, %value = tvm_ffi.pack.call @empty(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
                  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.pack.func @borrow(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
                  %count = tvm_ffi.pack.size %args : !tvm_ffi.pack.arguments
                  %one = arith.constant 1 : i64
                  %last = arith.subi %count, %one : i64
                  %value = tvm_ffi.pack.get %args[%last] : (!tvm_ffi.pack.arguments, i64) -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %value : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.pack.func @forward(%args: !tvm_ffi.pack.arguments) -> !tvm_ffi.any {
                  %ok, %value = tvm_ffi.pack.call @borrow(%args) : (!tvm_ffi.pack.arguments) -> (!tvm_ffi.bool, !tvm_ffi.any)
                  tvm_ffi.return %ok, %value : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @callee(%ok: !tvm_ffi.bool, %value: !tvm_ffi.tensor,
                                     %unused: !tvm_ffi.int) -> !tvm_ffi.tensor {
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
                                     %unused: !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
                  %call_ok, %call_value = tvm_ffi.call @callee(%ok, %value, %unused) :
                    (!tvm_ffi.bool, !tvm_ffi.tensor, !tvm_ffi.int) -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %result = tvm_ffi.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                  %payload = tvm_ffi.upcast %result : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_monad(%ok: !tvm_ffi.bool, %value: !tvm_ffi.tensor,
                                           %unused: !tvm_ffi.int) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
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
                tvm_ffi.func @empty() -> !tvm_ffi.none {
                  %none = tvm_ffi.constant.none
                  %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @invoke_empty() -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> {
                  %call_ok, %call_value = tvm_ffi.call @empty() : () -> (!tvm_ffi.bool, !tvm_ffi.any)
                  %result = tvm_ffi.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi.bool, !tvm_ffi.any) -> !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any>
                  %payload = tvm_ffi.upcast %result : !tvm_ffi.tuple<!tvm_ffi.bool, !tvm_ffi.any> -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)
            count, forward_empty, forward, invoke, invoke_monad, invoke_empty = (
                self._compile_entries(
                    module,
                    "count",
                    "forward_empty",
                    "forward",
                    "invoke",
                    "invoke_monad",
                    "invoke_empty",
                )
            )

        def check() -> None:
            tensor = torch.arange(4, device="cuda", dtype=torch.float32)
            for args in ((), (tensor,), (7, None, tensor)):
                self.assertEqual(count(*args), len(args))
            self.assertIsNone(forward_empty())
            self.assertIsInstance(forward_empty(1), Exception)
            for args in ((tensor,), (7, None, tensor)):
                borrowed = torch.from_dlpack(forward(*args))
                self.assertEqual(borrowed.data_ptr(), tensor.data_ptr())
                torch.testing.assert_close(borrowed, tensor)
            independent = torch.from_dlpack(forward(torch.arange(4, device="cuda")))
            torch.testing.assert_close(independent, torch.arange(4, device="cuda"))
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
