# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Execute local packed calls after monad and container lowering."""

import ast
import math
import unittest
from collections import Counter
from typing import TypeVar

import torch
import tvm_ffi
from trident.backend import TridentGraphModule
from trident.core import ir, register_all_passes
from trident.core.dialects.tvm_ffi import monad as tvm_ffi_monad
from trident.core.dialects.tvm_ffi import abi as tvm_ffi_abi
from trident.core.dialects.tvm_ffi import api as tvm_ffi_api
from trident.core.execution_engine import ExecutionEngine
from trident.core.passmanager import PassManager
from trident.ffi import Exception
from trident.guards.codes import ASTCode, GuardCode
from trident.guards.expressions.lowering import ExpressionContext
from trident.guards.expressions.syntax import parse_expression
from trident.guards.scope import Scope
from trident.guards.tree import Root
from trident.utils.monad import checked_call, pure
from trident.utils.runtime import find_runtime_libraries
from trident.utils.values import constant
from typing_extensions import override

from test.support.cuda import assert_no_cuda_leak
from test.support.ir import new_context

_T = TypeVar("_T")


class LocalCallTest(unittest.TestCase):
    @staticmethod
    def _compile_entries(
        module: ir.Module, *names: str
    ) -> tuple[tvm_ffi.Function, ...]:
        PassManager.parse("builtin.module(trident-lowering-pipeline)").run(
            module.operation
        )
        engine = ExecutionEngine(
            module, opt_level=3, shared_libs=find_runtime_libraries()
        )
        engine.initialize()
        return tuple(
            tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup(f"__tvm_ffi_{name}"), keep_alive_object=engine
            )
            for name in names
        )

    def test_dispatcher_outcomes(self) -> None:
        def dispatch(value: _T) -> _T:
            return value

        graph = TridentGraphModule(dispatch)
        # Execute the dispatcher directly so its sentinel does not trigger export.
        empty = graph.stub_compile()
        graph._output_spec = torch.utils._pytree.tree_structure(0)
        with graph.ctx:
            module = ir.Module.parse(
                """
                tvm_ffi_api.func @main_0(%value: !tvm_ffi_abi.int) -> !tvm_ffi_api.any {
                  %rejected = tvm_ffi_api.exception "GuardMatch" : !tvm_ffi_api.exception
                  %payload = tvm_ffi_api.upcast %rejected : !tvm_ffi_api.exception -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @main_1(%value: !tvm_ffi_abi.int) -> !tvm_ffi_api.any {
                  %payload = tvm_ffi_api.upcast %value : !tvm_ffi_abi.int -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @main_2(%value: !tvm_ffi_abi.int) -> !tvm_ffi_api.any {
                  %rejected = tvm_ffi_api.exception "GuardMatch" : !tvm_ffi_api.exception
                  %payload = tvm_ffi_api.upcast %rejected : !tvm_ffi_api.exception -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
            """
            )

        def add_guards(module: ir.Module, symbols: tuple[str, ...]) -> None:
            graph._dispatcher = Root()
            with graph.ctx, module.operation.location, ir.InsertionPoint(module.body):
                for function in list(module.body.operations):
                    if function.operation.name != "tvm_ffi_api.func":
                        continue
                    symbol = ir.StringAttr(function.attributes["sym_name"]).value
                    if symbol not in symbols:
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
                        symbol,
                    )

        graph._sub_modules = [
            module,
            *(ir.Module.create(loc=module.operation.location) for _ in range(3)),
        ]
        add_guards(module, ("main_0", "main_2"))
        rejected = graph.stub_compile()
        add_guards(module, ("main_1",))
        execute = graph.stub_compile()

        with graph.ctx:
            failure = ir.Module.parse(
                r"""
                llvm.mlir.global internal constant @error_kind("RuntimeError\00")
                llvm.mlir.global internal constant @error_message("dispatcher failure\00")
                llvm.func @TVMFFIErrorSetRaisedFromCStr(!llvm.ptr, !llvm.ptr)
                tvm_ffi_api.func @main_0(%value: !tvm_ffi_abi.int) -> !tvm_ffi_api.any {
                  %kind = llvm.mlir.addressof @error_kind : !llvm.ptr
                  %message = llvm.mlir.addressof @error_message : !llvm.ptr
                  llvm.call @TVMFFIErrorSetRaisedFromCStr(%kind, %message) : (!llvm.ptr, !llvm.ptr) -> ()
                  %none = tvm_ffi_abi.constant.none
                  %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                  %no = tvm_ffi_abi.constant.bool false
                  tvm_ffi_api.return %no, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
            """
            )
            _, success_function, *_ = module.body.operations
            success = success_function.operation.clone()
            failure.body.append(success)
        add_guards(failure, ("main_0", "main_1"))
        graph._sub_modules = [failure, ir.Module.create(loc=module.operation.location)]
        failed = graph.stub_compile()

        def check() -> None:
            self.assertIsInstance(empty(7), Exception)
            self.assertEqual(execute(7), 7)
            miss = rejected(7)
            self.assertIsInstance(miss, Exception)
            self.assertEqual(miss.get_kind(), "GuardMiss")
            self.assertEqual(execute(-3), -3)
            with self.assertRaisesRegex(RuntimeError, "dispatcher failure"):
                failed(7)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_erased_packed_boundaries(self) -> None:
        context = new_context()
        with context:
            register_all_passes()
            module = ir.Module.parse(
                """
                tvm_ffi_api.func @finite(%value: !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none>)
                    -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> {
                  %payload = tvm_ffi_api.upcast %value : !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.none> -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @string(%value: !tvm_ffi_api.str) -> !tvm_ffi_api.str {
                  %payload = tvm_ffi_api.upcast %value : !tvm_ffi_api.str -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @failed() -> !tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_abi.float> {
                  %none = tvm_ffi_abi.constant.none
                  %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                  %ok = tvm_ffi_abi.constant.bool false
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @invoke_failed() -> !tvm_ffi_abi.bool {
                  %ok, %value = tvm_ffi_api.call @failed() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
                  %payload = tvm_ffi_api.upcast %ok : !tvm_ffi_abi.bool -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
            """
            )
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
            module = ir.Module.parse(
                """
                tvm_ffi_api.pack.func @count(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_abi.int {
                  %count = tvm_ffi_api.pack.size %args : !tvm_ffi_abi.pack.arguments
                  %boxed = tvm_ffi_abi.to %count : i64 -> !tvm_ffi_abi.int
                  %value = tvm_ffi_api.upcast %boxed : !tvm_ffi_abi.int -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.pack.func @forward_empty(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_api.any {
                  %ok, %value = tvm_ffi_api.pack.call @empty(%args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
                  tvm_ffi_api.return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.pack.func @borrow(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_api.any {
                  %count = tvm_ffi_api.pack.size %args : !tvm_ffi_abi.pack.arguments
                  %one = arith.constant 1 : i64
                  %last = arith.subi %count, %one : i64
                  %value = tvm_ffi_api.pack.get %args[%last] : (!tvm_ffi_abi.pack.arguments, i64) -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.pack.func @forward(%args: !tvm_ffi_abi.pack.arguments) -> !tvm_ffi_api.any {
                  %ok, %value = tvm_ffi_api.pack.call @borrow(%args) : (!tvm_ffi_abi.pack.arguments) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
                  tvm_ffi_api.return %ok, %value : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @callee(%ok: !tvm_ffi_abi.bool, %value: !tvm_ffi_api.tensor,
                                     %unused: !tvm_ffi_abi.int) -> !tvm_ffi_api.tensor {
                  %success = tvm_ffi_api.get %ok : !tvm_ffi_abi.bool -> i1
                  %payload = scf.if %success -> (!tvm_ffi_api.any) {
                    %value_any = tvm_ffi_api.upcast %value : !tvm_ffi_api.tensor -> !tvm_ffi_api.any
                    scf.yield %value_any : !tvm_ffi_api.any
                  } else {
                    %none = tvm_ffi_abi.constant.none
                    %none_any = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                    scf.yield %none_any : !tvm_ffi_api.any
                  }
                  tvm_ffi_api.return %ok, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @invoke(%ok: !tvm_ffi_abi.bool, %value: !tvm_ffi_api.tensor,
                                     %unused: !tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> {
                  %call_ok, %call_value = tvm_ffi_api.call @callee(%ok, %value, %unused) :
                    (!tvm_ffi_abi.bool, !tvm_ffi_api.tensor, !tvm_ffi_abi.int) -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
                  %result = tvm_ffi_api.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
                  %payload = tvm_ffi_api.upcast %result : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @invoke_monad(%ok: !tvm_ffi_abi.bool, %value: !tvm_ffi_api.tensor,
                                           %unused: !tvm_ffi_abi.int) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> {
                  %call = tvm_ffi_monad.call @callee(%ok, %value, %unused) :
                    (!tvm_ffi_abi.bool, !tvm_ffi_api.tensor, !tvm_ffi_abi.int) -> !tvm_ffi_monad.result<!tvm_ffi_api.tensor>
                  %call_ok, %call_value = tvm_ffi_monad.unpack %call :
                    (!tvm_ffi_monad.result<!tvm_ffi_api.tensor>) -> (i1, !tvm_ffi_api.any)
                  %flag = tvm_ffi_abi.to %call_ok : i1 -> !tvm_ffi_abi.bool
                  %result = tvm_ffi_api.tuple.construct(%flag, %call_value) :
                    (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
                  %payload = tvm_ffi_api.upcast %result : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @empty() -> !tvm_ffi_abi.none {
                  %none = tvm_ffi_abi.constant.none
                  %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
                tvm_ffi_api.func @invoke_empty() -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> {
                  %call_ok, %call_value = tvm_ffi_api.call @empty() : () -> (!tvm_ffi_abi.bool, !tvm_ffi_api.any)
                  %result = tvm_ffi_api.tuple.construct(%call_ok, %call_value) :
                    (!tvm_ffi_abi.bool, !tvm_ffi_api.any) -> !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any>
                  %payload = tvm_ffi_api.upcast %result : !tvm_ffi_api.tuple<!tvm_ffi_abi.bool, !tvm_ffi_api.any> -> !tvm_ffi_api.any
                  %yes = tvm_ffi_abi.constant.bool true
                  tvm_ffi_api.return %yes, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                }
            """
            )
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
                        success, value = function(ok, tensor, 7)
                        self.assertEqual(success, ok)
                        if ok:
                            payload = torch.from_dlpack(value)
                            torch.testing.assert_close(payload, tensor)
                            self.assertEqual(payload.data_ptr(), tensor.data_ptr())
                        else:
                            self.assertIsNone(value)
            success, value = invoke_empty()
            self.assertTrue(success)
            self.assertIsNone(value)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_guard_expression_evaluation(self) -> None:
        cases = (
            ("and", "L['flag'] and L['xs'][0] == 7", False, False, True),
            ("comparison", "0 < len(L['xs']) < L['xs'][0]", False, False, True),
            ("conditional", "L['xs'][0] if L['flag'] else 42", False, 42, 7),
            ("empty", "not L['xs']", False, True, False),
            (
                "offset",
                "len(L['xs']) == 0 and L['unused'].storage_offset() == 0 and len(L['xs']) <= 2",
                False,
                True,
                False,
            ),
            (
                "offset_conditional",
                "L['xs'][0] == 7 if L['flag'] else L['unused'].storage_offset() + 1 != 2",
                False,
                True,
                True,
            ),
            (
                "offset_not",
                "not (L['unused'].storage_offset() == 0)",
                False,
                False,
                False,
            ),
            ("or", "L['flag'] or L['xs'][0] == 7", True, True, True),
        )
        with new_context() as context, ir.Location.unknown():
            register_all_passes()
            module = ir.Module.create()
            boolean = tvm_ffi_abi.BoolType.get()
            array = tvm_ffi_api.ArrayType.get(tvm_ffi_abi.IntType.get())
            any_type = tvm_ffi_api.AnyType.get()
            for name, text, *_ in cases:
                with ir.InsertionPoint(module.body):
                    function = tvm_ffi_api.FuncOp(
                        name,
                        ir.TypeAttr.get(
                            ir.FunctionType.get([boolean, array], [any_type])
                        ),
                    )
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    flag, xs = block.arguments
                    scope = Scope(flag, {"flag": Scope(flag), "xs": Scope(xs)})
                    expression = parse_expression(ast.parse(text, mode="eval").body)
                    result = expression.build(ExpressionContext(text, scope, context))
                    success, payload = tvm_ffi_monad.unpack(result)
                    tvm_ffi_api.return_(tvm_ffi_abi.to(success), payload)
            self.assertTrue(module.operation.verify())
            functions = self._compile_entries(module, *(name for name, *_ in cases))

        def check() -> None:
            for function, (
                name,
                _,
                empty_flag,
                empty_expected,
                populated_expected,
            ) in zip(functions, cases, strict=True):
                with self.subTest(case=name):
                    empty = function(empty_flag, [])
                    populated = function(name != "or", [7])
                    self.assertEqual(empty, empty_expected)
                    self.assertIs(type(empty), type(empty_expected))
                    self.assertEqual(populated, populated_expected)
                    self.assertIs(type(populated), type(populated_expected))

        assert_no_cuda_leak(
            self, check, torch.device("cuda", torch.cuda.current_device())
        )

    def test_scalar_predicates(self) -> None:
        # Runtime inputs keep IEEE truth checks observable through Guard lowering.
        expressions = (
            ("real_not", "not L['real']"),
            ("real_truth", "7 if L['real'] else 9"),
        )
        with new_context() as context, ir.Location.unknown():
            register_all_passes()
            module = ir.Module.create()
            for name, text in expressions:
                with ir.InsertionPoint(module.body):
                    function = tvm_ffi_api.FuncOp(
                        name,
                        ir.TypeAttr.get(
                            ir.FunctionType.get(
                                [tvm_ffi_abi.FloatType.get()],
                                [tvm_ffi_api.AnyType.get()],
                            )
                        ),
                    )
                with ir.InsertionPoint(function.add_entry_block()) as ip:
                    [real] = ip.block.arguments
                    scope = Scope(real, {"real": Scope(real)})
                    expression = parse_expression(ast.parse(text, mode="eval").body)
                    result = expression.build(ExpressionContext(text, scope, context))
                    success, payload = tvm_ffi_monad.unpack(result)
                    tvm_ffi_api.return_(tvm_ffi_abi.to(success), payload)
            self.assertTrue(module.operation.verify())
            functions = self._compile_entries(
                module, *(name for name, _ in expressions)
            )

        def check() -> None:
            for real in (-0.0, 2.5, math.nan):
                for function, (name, _), expected in zip(
                    functions, expressions, (not real, 7 if real else 9), strict=True
                ):
                    with self.subTest(case=name, real=real):
                        actual = function(real)
                        self.assertIs(type(actual), type(expected))
                        self.assertEqual(actual, expected)

        assert_no_cuda_leak(
            self, check, torch.device("cuda", torch.cuda.current_device())
        )

    def test_shared_prefix_dispatch(self) -> None:
        counts: Counter[int] = Counter()

        def record(kind: int) -> None:
            counts[kind] += 1

        tvm_ffi.register_global_func("trident.test.guard.record", record, override=True)

        class RuntimeCode(GuardCode):
            @override
            def build_result(self, scope: Scope, context: ir.Context) -> list[ir.Value]:
                if self.text == "failure":
                    return [
                        tvm_ffi_monad.call(
                            tvm_ffi_monad.ResultType.get(tvm_ffi_abi.BoolType.get()),
                            ir.FlatSymbolRefAttr.get("guard_failure"),
                            [],
                        )
                    ]
                return [pure(constant(self.text != "rejection"))]

        def dispatch(value: int) -> int:
            return value

        graph = TridentGraphModule(dispatch)
        graph._output_spec = torch.utils._pytree.tree_structure(0)

        def build(
            *,
            failing: bool = False,
            paths: tuple[tuple[str, ...], ...] | None = None,
            repeated: str | None = None,
        ) -> tvm_ffi.Function:
            with graph.ctx, ir.Location.unknown():
                module = (
                    ir.Module.parse(
                        r"""
                    llvm.mlir.global internal constant @error_kind("RuntimeError\00")
                    llvm.mlir.global internal constant @error_message("guard failure\00")
                    llvm.func @TVMFFIErrorSetRaisedFromCStr(!llvm.ptr, !llvm.ptr)
                    tvm_ffi_api.func private @guard_failure() -> !tvm_ffi_abi.bool {
                      %kind = llvm.mlir.addressof @error_kind : !llvm.ptr
                      %message = llvm.mlir.addressof @error_message : !llvm.ptr
                      llvm.call @TVMFFIErrorSetRaisedFromCStr(%kind, %message) : (!llvm.ptr, !llvm.ptr) -> ()
                      %none = tvm_ffi_abi.constant.none
                      %payload = tvm_ffi_api.upcast %none : !tvm_ffi_abi.none -> !tvm_ffi_api.any
                      %no = tvm_ffi_abi.constant.bool false
                      tvm_ffi_api.return %no, %payload : !tvm_ffi_abi.bool, !tvm_ffi_api.any
                    }
                """
                    )
                    if failing
                    else ir.Module.create()
                )
                any_type = tvm_ffi_api.AnyType.get()
                type_ = tvm_ffi_abi.IntType.get()
                graph._dispatcher = Root()
                with ir.InsertionPoint(module.body):
                    for index in range(3):
                        input_types = {"value": type_}
                        texts = ["L['value'] >= 0", f"L['value'] == {index}"]
                        if paths is not None:
                            texts = list(paths[index])
                        codes: list[GuardCode] = []
                        for text in texts:
                            code = ASTCode.parse(text, None)
                            assert code is not None
                            codes.append(code)
                        if repeated is not None and index < 2:
                            codes.insert(0, RuntimeCode(repeated, None))
                        if failing and index == 1:
                            codes.insert(0, RuntimeCode("failure", None))
                        graph._dispatcher.add(
                            codes,
                            input_types,
                            f"main_{index}",
                        )
                        function = tvm_ffi_api.FuncOp(
                            f"main_{index}",
                            ir.TypeAttr.get(ir.FunctionType.get([type_], [any_type])),
                        )
                        entry = function.add_entry_block()
                        with ir.InsertionPoint(entry):
                            checked_call(
                                "trident.test.guard.record",
                                [constant(100 + index)],
                                tvm_ffi_abi.NoneType.get(),
                            )
                            result = pure(tvm_ffi_api.upcast(any_type, constant(index)))
                            success, payload = tvm_ffi_monad.unpack(result)
                            tvm_ffi_api.return_(tvm_ffi_abi.to(success), payload)
                graph._sub_modules = [
                    module,
                    ir.Module.create(),
                    ir.Module.create(),
                ]
                graph._build_dispatcher(module)
                [dispatch] = self._compile_entries(module, "dispatch")
                return dispatch

        failed = build(failing=True)
        repeated_rejection = build(repeated="rejection")
        nonadjacent = build(
            paths=(
                ("L['value'] >= 0", "L['value'] < 1"),
                ("L['value'] <= 2",),
                ("L['value'] >= 0", "L['value'] < 3"),
            )
        )
        terminal = build(paths=((), ("L['value'] >= 0",), ()))
        shorter = build(
            paths=(
                ("L['value'] >= 0", "L['value'] < 1"),
                ("L['value'] >= 0",),
                ("L['value'] >= 0", "L['value'] < 3"),
            )
        )

        def check() -> None:
            self.assertEqual(nonadjacent(2), 1)
            self.assertEqual(terminal(2), 0)
            self.assertEqual(shorter(2), 1)
            miss = shorter(-1)
            self.assertIsInstance(miss, Exception)
            self.assertEqual(miss.get_kind(), "GuardMiss")
            counts.clear()
            self.assertEqual(repeated_rejection(2), 2)
            self.assertEqual((counts[100], counts[101], counts[102]), (0, 0, 1))
            counts.clear()
            with self.assertRaisesRegex(RuntimeError, "guard failure"):
                failed(2)
            self.assertFalse(counts)
            # An earlier match must suppress later fallible checks.
            self.assertEqual(failed(0), 0)
            self.assertEqual(counts[100], 1)

        assert_no_cuda_leak(
            self, check, torch.device("cuda", torch.cuda.current_device())
        )
