# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Execute local packed calls after monad and container lowering."""

import unittest
from collections import Counter
from unittest.mock import patch

import torch
import tvm_ffi
from trident import capi_utils
from trident.backend import TridentGraphModule
from trident.core import ir, register_all_passes
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.core.execution_engine import ExecutionEngine
from trident.core.passmanager import PassManager
from trident.ffi import Exception
from trident.guards.codes import ASTCode, GuardCode
from trident.guards.scope import Scope
from trident.guards.tree import Root
from trident.ir import checked_call, constant, pure

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
        empty = graph.stub_compile()
        graph._output_spec = torch.utils._pytree.tree_structure(0)
        with graph.ctx:
            module = ir.Module.parse("""
                tvm_ffi.func @main_0(%value: !tvm_ffi.tensor) -> !tvm_ffi.any {
                  %rejected = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
                  %payload = tvm_ffi.upcast %rejected : !tvm_ffi.exception -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @main_1(%value: !tvm_ffi.int) -> !tvm_ffi.any {
                  %payload = tvm_ffi.upcast %value : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @main_2() -> !tvm_ffi.any {
                  %answer = tvm_ffi.constant.int 42
                  %payload = tvm_ffi.upcast %answer : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
                tvm_ffi.func @main_3(%first: !tvm_ffi.int, %second: !tvm_ffi.int) -> !tvm_ffi.any {
                  %payload = tvm_ffi.upcast %first : !tvm_ffi.int -> !tvm_ffi.any
                  %yes = tvm_ffi.constant.bool true
                  tvm_ffi.return %yes, %payload : !tvm_ffi.bool, !tvm_ffi.any
                }
            """)

        def add_guards(module: ir.Module) -> None:
            graph._dispatcher = Root()
            with graph.ctx, module.operation.location, ir.InsertionPoint(module.body):
                for function in list(module.body.operations):
                    if function.operation.name != "tvm_ffi.func":
                        continue
                    input_types = {
                        f"arg{index}": type_
                        for index, type_ in enumerate(
                            function.function_type.value.inputs
                        )
                    }
                    graph._dispatcher.add(
                        (),
                        input_types,
                        tuple(
                            (name,)
                            for name in (
                                reversed(input_types)
                                if graph._dispatcher.size == 3
                                else input_types
                            )
                        ),
                        ir.StringAttr(function.attributes["sym_name"]).value,
                    )

        add_guards(module)
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
                tvm_ffi.func @main_0(%value: !tvm_ffi.any) -> !tvm_ffi.any {
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
        add_guards(failure)
        graph._sub_modules = [failure, ir.Module.create(loc=module.operation.location)]
        failed = graph.stub_compile()

        def check() -> None:
            self.assertIsInstance(empty(7), Exception)
            self.assertEqual(execute(), 42)
            self.assertEqual(execute(7, 8), 8)
            self.assertIsInstance(execute(7, 8, 9), Exception)
            self.assertIsInstance(execute(7, "wrong type"), Exception)
            self.assertEqual(execute(7), 7)
            self.assertIsInstance(execute("all rejected"), Exception)
            tensor = torch.ones(2, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "GuardMatch"):
                execute(tensor)
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

    def test_independent_dispatch_checks(self) -> None:
        counts: Counter[int] = Counter()

        def record(kind: int) -> None:
            counts[kind] += 1

        tvm_ffi.register_global_func("trident.test.guard.record", record, override=True)

        mutation_inputs: list[torch.Tensor] = []

        def mutate(value: tvm_ffi.Tensor) -> None:
            # DLPack borrows the source TensorImpl's shape storage. Keep the
            # rank and backing allocation fixed while changing its size.
            self.assertEqual(tuple(value.shape), (1,))
            mutation_inputs[0].resize_(2)
            self.assertEqual(tuple(value.shape), (2,))

        tvm_ffi.register_global_func("trident.test.guard.mutate", mutate, override=True)

        class RuntimeCode(GuardCode):
            def build_result(self, scope: Scope, context: ir.Context) -> list[ir.Value]:
                if self.text == "failure":
                    return [
                        tvm_ffi_d.monad_call(
                            tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get()),
                            ir.FlatSymbolRefAttr.get("guard_failure"),
                            [],
                        )
                    ]
                if self.text == "mutation":
                    value = scope["value"].value
                    checked_call(
                        "trident.test.guard.mutate", [value], tvm_ffi_d.NoneType.get()
                    )
                    return [pure(constant(False))]
                checked_call(
                    "trident.test.guard.record", [constant(9)], tvm_ffi_d.NoneType.get()
                )
                return [pure(constant(self.text != "rejection"))]

        def dispatch(value: int) -> int:
            return value

        graph = TridentGraphModule(dispatch)
        graph._output_spec = torch.utils._pytree.tree_structure(0)

        def build(
            *,
            barrier: bool = False,
            overlap: bool = False,
            failing: bool = False,
            guard_mutation: bool = False,
            repeated: str | None = None,
        ) -> tvm_ffi.Function:
            with graph.ctx, ir.Location.unknown():
                module = (
                    ir.Module.parse(r"""
                    llvm.mlir.global internal constant @error_kind("RuntimeError\00")
                    llvm.mlir.global internal constant @error_message("guard failure\00")
                    llvm.func @TVMFFIErrorSetRaisedFromCStr(!llvm.ptr, !llvm.ptr)
                    tvm_ffi.func private @guard_failure() -> !tvm_ffi.bool {
                      %kind = llvm.mlir.addressof @error_kind : !llvm.ptr
                      %message = llvm.mlir.addressof @error_message : !llvm.ptr
                      llvm.call @TVMFFIErrorSetRaisedFromCStr(%kind, %message) : (!llvm.ptr, !llvm.ptr) -> ()
                      %none = tvm_ffi.constant.none
                      %payload = tvm_ffi.upcast %none : !tvm_ffi.none -> !tvm_ffi.any
                      %no = tvm_ffi.constant.bool false
                      tvm_ffi.return %no, %payload : !tvm_ffi.bool, !tvm_ffi.any
                    }
                """)
                    if failing
                    else ir.Module.create()
                )
                any_type = tvm_ffi_d.AnyType.get()
                type_ = (
                    tvm_ffi_d.TensorType.get()
                    if guard_mutation
                    else tvm_ffi_d.IntType.get()
                )
                graph._dispatcher = Root()
                with ir.InsertionPoint(module.body):
                    for index in range(3):
                        input_types = {"value": type_}
                        texts = ["L['value'] >= 0", f"L['value'] == {index}"]
                        if guard_mutation:
                            texts = [
                                "L['value'].ndimension() == 1",
                                "L['value'].size()[0] >= 1",
                                f"L['value'].size()[0] == {1 if index < 2 else 2}",
                            ]
                        if overlap and index == 1:
                            texts[1] = "L['value'] <= 2"
                        codes: list[GuardCode] = []
                        for text in texts:
                            code = ASTCode.parse(text, None)
                            assert code is not None
                            codes.append(code)
                        if repeated is not None and index < 2:
                            codes.insert(0, RuntimeCode(repeated, None))
                        if barrier and index == 1:
                            codes.insert(0, RuntimeCode("barrier", None))
                        if failing and index == 1:
                            codes.insert(0, RuntimeCode("failure", None))
                        if guard_mutation and index == 0:
                            codes.append(RuntimeCode("mutation", None))
                        graph._dispatcher.add(
                            codes,
                            input_types,
                            (("value",),),
                            f"main_{index}",
                        )
                        function = tvm_ffi_d.FuncOp(
                            f"main_{index}",
                            ir.TypeAttr.get(ir.FunctionType.get([type_], [any_type])),
                        )
                        entry = function.add_entry_block()
                        with ir.InsertionPoint(entry):
                            checked_call(
                                "trident.test.guard.record",
                                [constant(100 + index)],
                                tvm_ffi_d.NoneType.get(),
                            )
                            result = pure(tvm_ffi_d.upcast(any_type, constant(index)))
                            success, payload = tvm_ffi_d.monad_unpack(result)
                            tvm_ffi_d.return_(tvm_ffi_d.to(success), payload)
                graph._sub_modules = [
                    module,
                    ir.Module.create(),
                    ir.Module.create(),
                ]
                graph._build_dispatcher(module)
                return self._compile_entries(module, "dispatch")[0]

        effectful = build(barrier=True)
        overlapping = build(overlap=True)
        failed = build(failing=True)
        guard_mutating = build(guard_mutation=True)
        repeated_rejection = build(repeated="rejection")

        def check() -> None:
            self.assertEqual(overlapping(2), 1)
            counts.clear()
            self.assertEqual(repeated_rejection(2), 2)
            self.assertEqual(counts[9], 2)
            self.assertEqual((counts[100], counts[101], counts[102]), (0, 0, 1))
            counts.clear()
            with self.assertRaisesRegex(RuntimeError, "guard failure"):
                failed(2)
            counts.clear()
            self.assertEqual(effectful(2), 2)
            self.assertEqual(counts[9], 1)
            # An earlier match must suppress later effectful/fallible checks.
            for function in (effectful, failed):
                counts.clear()
                self.assertEqual(function(0), 0)
                self.assertEqual(counts[9], 0)
                self.assertEqual(counts[100], 1)
            counts.clear()
            value = torch.ones(2, device="cuda").resize_(1)
            mutation_inputs.append(value)
            try:
                self.assertEqual(guard_mutating(tvm_ffi.from_dlpack(value)), 2)
                self.assertEqual(tuple(value.shape), (2,))
                self.assertEqual(tuple(counts[100 + i] for i in range(3)), (0, 0, 1))
            finally:
                mutation_inputs.clear()

        assert_no_cuda_leak(
            self, check, torch.device("cuda", torch.cuda.current_device())
        )
