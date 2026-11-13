# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Focused tests for translating Dynamo guards to Trident guard IR."""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass

from torch._guards import GuardSource
from trident.core import ir
from trident.core.dialects import func
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.guards import parse_guards
from trident.guards.codes import (
    ASTCode,
    DynamoAttributeAbsentCode,
    GuardCode,
    RequiresGradCode,
    TensorDeviceCode,
    TensorDTypeCode,
    TensorRankCode,
    TypeIdCode,
)
from trident.guards.expressions import (
    ExpressionBuilder,
    UnsupportedExpression,
    parse_expression,
)
from trident.guards.scope import Scope
from trident.guards.source import parse_source
from trident.guards.tree import (
    ExecutionNode,
    GuardNode,
    InputNode,
    Node,
    Root,
    build_dispatcher,
)
from trident.ir import constant

from test.support.ir import new_context


@dataclass(frozen=True)
class FakeGuard:
    create_fn: str
    code_list: list[str] | None
    name: str = ""
    source: GuardSource = GuardSource.LOCAL

    def create_fn_name(self) -> str:
        return self.create_fn


class GuardExpressionTest(unittest.TestCase):
    """IR generation, invalid comparisons and conservative view guards."""

    def _build_expression(self, text: str) -> tuple[ir.Type, str]:
        """Insert a guard expression into a textually parsed IR fixture."""
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.parse(
                """
                module {
                  func.func @guard_expression() {
                    return
                  }
                }
                """
            )
            [function] = module.body.operations
            [region] = function.regions
            [block] = region.blocks
            *_, terminator = block.operations
            with ir.InsertionPoint(terminator):
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                result = ExpressionBuilder(text, None, context).build(code.expression)
            self.assertTrue(module.operation.verify())
            result_type = result.type.payload_type
            module_text = str(module)
        return result_type, module_text

    def test_comparison_reads_and_compatibility_prefix(self) -> None:
        cases = (
            ("0 <= len(L['xs']) <= 2", 1),
            (
                "len(L['xs']) == 0 and L['unused'].storage_offset() == 0 and L['missing'] > 0",
                1,
            ),
            ("len(L['xs']) == len(L['xs']) < L['unused'].storage_offset()", 3),
        )
        for text, reads in cases:
            with (
                self.subTest(text=text),
                new_context() as context,
                ir.Location.unknown(),
            ):
                tuple_type = tvm_ffi_d.TupleType.get([tvm_ffi_d.IntType.get()])
                result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp("guard", ([tuple_type], [result_type]))
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    [argument] = block.arguments
                    scope = Scope(argument, {"xs": Scope(argument)})
                    code = ASTCode.parse(text, None)
                    assert code is not None
                    func.return_(code.build_result(scope, context))
                self.assertTrue(module.operation.verify())
                self.assertEqual(str(module).count('"ffi.ArraySize"'), reads)
                self.assertNotIn("tensor.storage_offset", str(module))

    def test_container_length_extracts_nested_input(self) -> None:
        for container in (tvm_ffi_d.ArrayType, tvm_ffi_d.TupleType):
            with (
                self.subTest(container=container.__name__),
                new_context() as context,
                ir.Location.unknown(),
            ):
                tensor = tvm_ffi_d.TensorType.get()
                type_ = (
                    container.get(tensor)
                    if container is tvm_ffi_d.ArrayType
                    else container.get([tensor, tensor])
                )
                array = tvm_ffi_d.ArrayType.get(type_)
                text = "len(L['xs'][0]) == 2"
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                module = ir.Module.create()
                result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp("guard", ([array], [result_type]))
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    [argument] = block.arguments
                    scope = Scope(argument, {"xs": Scope(argument)})
                    results = code.build_result(scope, context)
                    func.return_(results)
                self.assertEqual([result.type for result in results], [result_type])
                self.assertTrue(module.operation.verify())
                module_text = f"{module}"
                self.assertIn("tvm_ffi.array.get", module_text)
                self.assertIn("tvm_ffi.monad.bind", module_text)
                self.assertIn("arith.cmpi eq", module_text)
                # Tuple guards must check runtime arity instead of trusting the schema.
                if container is tvm_ffi_d.TupleType:
                    self.assertIn('"ffi.ArraySize"', module_text)
                else:
                    self.assertIn("tvm_ffi.array.size", module_text)

    def test_input_reads_reuse_local_bindings(self) -> None:
        for text in (
            "L['xs'][0] + L['xs'][0] == 2",
            "len(L['xs']) == L['xs'][0] == L['xs'][0]",
        ):
            with (
                self.subTest(text=text),
                new_context() as context,
                ir.Location.unknown(),
            ):
                integer = tvm_ffi_d.IntType.get()
                array = tvm_ffi_d.ArrayType.get(integer)
                result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp("guard", ([array], [result_type]))
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    [argument] = block.arguments
                    root = Scope(argument, {"xs": Scope(argument)})
                    builder = ExpressionBuilder(text, root, context)
                    results = [
                        builder.build(
                            parse_expression(ast.parse(text, mode="eval").body)
                        )
                    ]
                    # Values produced in bind bodies must not escape via the cache.
                    builder.build(
                        parse_expression(ast.parse("L['xs'][0]", mode="eval").body)
                    )
                    func.return_(results)
                self.assertIsNone(root["xs"].get(0))
                self.assertTrue(module.operation.verify())
                if text.startswith("L["):
                    self.assertEqual(str(module).count("tvm_ffi.array.get"), 1)

    def test_input_scope_restored_after_construction_failure(self) -> None:
        # The second operand fails inside its bind while the outer bind is entering.
        text = "len(L['xs']) + len(L['xs'][0])"
        with new_context() as context, ir.Location.unknown():
            array = tvm_ffi_d.ArrayType.get(tvm_ffi_d.IntType.get())
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("guard", ([array], []))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                [argument] = block.arguments
                root = Scope(argument, {"xs": Scope(argument)})
                builder = ExpressionBuilder(text, root, context)
                expression = parse_expression(ast.parse(text, mode="eval").body)
                with self.assertRaisesRegex(AssertionError, "len requires a container"):
                    builder.build(expression)
                self.assertIs(builder.scope, root)
                self.assertIsNone(root["xs"].get(0))
                self.assertEqual(ir.InsertionPoint.current.block, block)
            # A failed construction leaves incomplete IR; discard the module.

    def test_invalid_comparisons(self) -> None:
        for text, message in (
            ("L['x'] is L['y'] is L['z']", "identity guard comparison must be binary"),
            ("1 in (1, 2)", "unsupported guard comparison operator"),
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(UnsupportedExpression, message),
            ):
                self._build_expression(text)

    def test_supported_expressions(self) -> None:
        cases = (
            ("bitwise_or", "(1 | 2) == 3", tvm_ffi_d.BoolType, ("arith.ori",)),
            (
                "boolean_logic",
                "True and not False",
                tvm_ffi_d.BoolType,
                ("arith.cmpi eq",),
            ),
            ("boolean_or", "True | False", tvm_ffi_d.BoolType, ("arith.ori",)),
            (
                "chained_comparison",
                "1 == 1 <= 3",
                tvm_ffi_d.BoolType,
                ("arith.cmpi eq", "arith.cmpi sle"),
            ),
            (
                "dtype",
                "torch.float32",
                tvm_ffi_d.DTypeType,
                ("tvm_ffi.constant.dtype",),
            ),
            (
                "float_comparison",
                "1.0 + 2.0 <= 4.0",
                tvm_ffi_d.BoolType,
                ("arith.addf", "arith.cmpf"),
            ),
            ("float_negation", "-1.5", tvm_ffi_d.FloatType, ("arith.negf",)),
            (
                "float_not_negative_zero",
                "not -0.0",
                tvm_ffi_d.BoolType,
                ("arith.cmpf oeq",),
            ),
            ("integer_addition", "1 + 2", tvm_ffi_d.IntType, ("arith.addi",)),
            ("integer_not", "not 0", tvm_ffi_d.BoolType, ("arith.cmpi eq",)),
            (
                "mixed_boolean_float",
                "True + 2.0",
                tvm_ffi_d.FloatType,
                ("arith.uitofp", "arith.addf"),
            ),
            (
                "mixed_boolean_integer",
                "True | 1",
                tvm_ffi_d.IntType,
                (
                    "arith.extui",
                    "arith.ori",
                ),
            ),
            ("mixed_select", "1 if True else 'other'", tvm_ffi_d.AnyType, ("scf.if",)),
            ("negation", "-1", tvm_ffi_d.IntType, ("arith.subi",)),
            ("operand_or", "0 or 2.5", tvm_ffi_d.AnyType, ("scf.if",)),
            ("select", "1 if True else 2", tvm_ffi_d.IntType, ("scf.if",)),
            (
                "string_equality",
                "'name' == 'name'",
                tvm_ffi_d.BoolType,
                ("tvm_ffi.eq", "tvm_ffi.to"),
            ),
            (
                "true_division",
                "5 / 2",
                tvm_ffi_d.FloatType,
                ("arith.divf",),
            ),
        )
        for name, text, type_class, operations in cases:
            with self.subTest(case=name):
                actual_type, module = self._build_expression(text)
                self.assertIsInstance(actual_type, type_class)
                for operation in operations:
                    self.assertIn(operation, module)
                self.assertNotIn("arith.constant", module)
                self.assertNotIn("torch.aten.", module)

    def test_tensor_base_fallback(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "Tensor\\._base guard"):
            result_type, module = self._build_expression("L['x']._base.size()[0] == 1")
        self.assertIsInstance(result_type, tvm_ffi_d.BoolType)
        self.assertIn("tvm_ffi.constant.bool true", module)


class DispatcherTreeTest(unittest.TestCase):
    """Independent candidates and module-local SSA lifetimes."""

    def _build_dispatcher(self, dispatcher: Root) -> ir.Module:
        module = ir.Module.create()
        any_type = tvm_ffi_d.AnyType.get()

        def add_functions(node: Node) -> None:
            if isinstance(node, ExecutionNode):
                graph_input_types = node.graph_input_types
                function = tvm_ffi_d.FuncOp(
                    node.graph_symbol,
                    ir.TypeAttr.get(
                        ir.FunctionType.get(
                            graph_input_types,
                            [any_type],
                        )
                    ),
                    sym_visibility="private",
                )
                with ir.InsertionPoint(function.add_entry_block()):
                    tvm_ffi_d.return_(
                        constant(True), tvm_ffi_d.upcast(any_type, constant(node.index))
                    )
            else:
                assert isinstance(node, (GuardNode, InputNode, Root))
                for child in node.children:
                    add_functions(child)

        with ir.InsertionPoint(module.body):
            add_functions(dispatcher)
            function = tvm_ffi_d.PackFuncOp(
                "dispatch",
                ir.TypeAttr.get(
                    ir.FunctionType.get([tvm_ffi_d.PackArgumentsType.get()], [any_type])
                ),
            )
            with ir.InsertionPoint(function.add_entry_block()) as ip:
                [arguments] = ip.block.arguments
                success, payload = build_dispatcher(dispatcher, Scope(arguments))
                tvm_ffi_d.return_(success, payload)
        self.assertTrue(module.operation.verify())
        return module

    def test_ordered_prefix_and_fresh_modules(self) -> None:
        with new_context(), ir.Location.unknown():
            input_types = {"x": tvm_ffi_d.IntType.get()}
            checks = [["L['x'] >= 0", f"L['x'] == {index}"] for index in range(3)]
            dispatcher = Root()
            for index, texts in enumerate(checks):
                dispatcher.add(
                    parse_guards([FakeGuard("SHAPE_ENV", texts)]),
                    input_types,
                    (("x",),),
                    f"run_{index}",
                )
            self.assertEqual(len(dispatcher.children), 3)
            # Reuse recipes and the planned tree, but rebuild all SSA in each module.
            for _ in range(2):
                module = self._build_dispatcher(dispatcher)
                self.assertEqual(f"{module}".count("arith.cmpi sge"), 3)
            previous = f"{module}"
            dispatcher.add(
                parse_guards([FakeGuard("SHAPE_ENV", ["L['x'] >= 0", "L['x'] == 3"])]),
                input_types,
                (("x",),),
                "run_3",
            )
            updated = self._build_dispatcher(dispatcher)
            self.assertNotIn("@run_3", previous)
            self.assertIn("tvm_ffi.monad.call @run_3", f"{updated}")
            self.assertEqual(f"{updated}".count("arith.cmpi sge"), 4)
            self.assertEqual(f"{module}", previous)
            self.assertTrue(module.operation.verify())


class GuardParsingTest(unittest.TestCase):
    """Content-based parsing, code composition, metadata and rejected inputs."""

    def _assert_parsed(
        self,
        guard: FakeGuard,
        code_types: tuple[type[GuardCode], ...],
    ) -> tuple[GuardCode, ...]:
        parsed = parse_guards([guard])
        self.assertEqual(tuple(type(item) for item in parsed), code_types)
        source = parse_source(guard.name)
        for code in parsed:
            self.assertEqual(
                None if code.source is None else ast.dump(code.source),
                None if source is None else ast.dump(source),
            )
        return parsed

    def test_code_ordering_and_deduplication(self) -> None:
        general = "L['x'] >= 0"
        rank = "L['x'].ndimension() == 2"
        scalar = "L['n'] == 3"
        other = "L['x'].size()[0] >= 2"
        codes = parse_guards(
            [
                FakeGuard("UNKNOWN", [general, rank, scalar, other], "L['x']"),
                FakeGuard("CONSTANT_MATCH", [scalar], "L['n']"),
                FakeGuard("SHAPE_ENV", ['L["n"] == 3']),
            ]
        )
        self.assertEqual(
            tuple(type(code) for code in codes),
            (TensorRankCode, ASTCode, ASTCode, ASTCode),
        )
        self.assertEqual(tuple(code.priority for code in codes), (1, 1, 2, 2))
        _, _, *remaining = codes
        self.assertEqual(tuple(code.text for code in remaining), (general, other))

    def test_constant_values(self) -> None:
        supported = (
            "L['value'] is None",
            "L['value'] == True",
            "L['value'] == -7",
            "L['value'] == 1.5",
            "L['value'] == 'constant'",
            "L['value'] == device(type='cuda', index=1)",
            "L['value'] == torch.float32",
        )
        for text in supported:
            with self.subTest(text=text):
                parsed = self._assert_parsed(
                    FakeGuard("CONSTANT_MATCH", [text], "L['value']"),
                    (ASTCode,),
                )
                [code] = parsed
                assert isinstance(code, ASTCode)
                self.assertEqual(
                    code.expression,
                    parse_expression(ast.parse(text, mode="eval").body),
                )

    def test_invalid_device(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            parse_guards(
                [
                    FakeGuard(
                        "TENSOR_MATCH",
                        ["str(L['x'].device) == 'not-a-device'"],
                        "L['x']",
                    )
                ]
            )

    def test_rejected_guards(self) -> None:
        cases = (
            (
                "constant_device_call",
                FakeGuard(
                    "CONSTANT_MATCH", ["L['value'] == device('cpu')"], "L['value']"
                ),
            ),
            (
                "constant_global_source",
                FakeGuard("CONSTANT_MATCH", ["G['value'] == 1"], "L['value']"),
            ),
            (
                "constant_list",
                FakeGuard("CONSTANT_MATCH", ["L['value'] == [1, 2]"], "L['value']"),
            ),
            ("constant_missing_code", FakeGuard("CONSTANT_MATCH", None, "L['x']")),
            (
                "mixed_unsupported_code",
                FakeGuard("UNKNOWN", ["L['x'] >= 0", "unsupported(L['x'])"], "L['x']"),
            ),
            (
                "shape_invalid_type_template",
                FakeGuard(
                    "SEQUENCE_LENGTH",
                    [
                        "len(L['x'].shape) == 2",
                        "___check_type_id(L['y'].shape, 1), type=<class 'torch.Size'>",
                    ],
                    "L['x'].shape",
                ),
            ),
            (
                "shape_unsupported_code",
                FakeGuard(
                    "SEQUENCE_LENGTH",
                    ["len(L['x'].shape) == 2", "unsupported(L['x'])"],
                    "L['x'].shape",
                ),
            ),
            (
                "type_match_object_identity",
                FakeGuard("TYPE_MATCH", ["___check_obj_id(L['x'], 1)"], "L['x']"),
            ),
            ("unknown_kind", FakeGuard("UNKNOWN", None)),
        )
        for name, guard in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(
                    AssertionError, "unsupported Dynamo guard"
                ) as caught:
                    parse_guards([guard])
                self.assertIn(guard.create_fn, str(caught.exception))
                self.assertIn(repr(guard.name), str(caught.exception))
                if guard.code_list:
                    *_, last_code = guard.code_list
                    self.assertIn(repr(last_code), str(caught.exception))

    def test_shape_source(self) -> None:
        self._assert_parsed(
            FakeGuard(
                "SEQUENCE_LENGTH",
                [
                    "___check_type_id(L['x'].shape, 1), type=<class 'torch.Size'>",
                    "len(L['x'].shape) == 2",
                ],
                "L['x'].shape",
            ),
            (TypeIdCode, ASTCode),
        )

    def test_supported_expressions(self) -> None:
        cases = (
            (
                FakeGuard("CONSTANT_MATCH", ["L['x'] == 1", "L['x'] == 2"], "L['x']"),
                (ASTCode, ASTCode),
            ),
            (
                FakeGuard("CONSTANT_MATCH", ["L['value'] != 1"], "L['value']"),
                (ASTCode,),
            ),
            (
                FakeGuard("CONSTANT_MATCH", ["1 == L['value']"], "L['value']"),
                (ASTCode,),
            ),
            (
                FakeGuard("DUPLICATE_INPUT", ["L['x'] is L['y']"], "L['x']"),
                (ASTCode,),
            ),
            (FakeGuard("GLOBAL_STATE", None), ()),
            (
                FakeGuard(
                    "SEQUENCE_LENGTH",
                    [
                        "___check_type_id(L['xs'], 1), type=<class 'list'>",
                        "len(L['xs']) == 2",
                    ],
                    "L['xs']",
                ),
                (TypeIdCode, ASTCode),
            ),
            (
                FakeGuard("SHAPE_ENV", ["L['x'].size()[0] == 2"], "L['x']"),
                (ASTCode,),
            ),
            (FakeGuard("SHAPE_ENV", None), ()),
            (
                FakeGuard(
                    "SHAPE_ENV",
                    [
                        "2 <= L['x'].size()[0] <= 8",
                        "L['x'].size()[1] == 4 * L['y'].size()[1]",
                    ],
                ),
                (ASTCode, ASTCode),
            ),
            (
                FakeGuard("TENSOR_MATCH", ["L['x'].ndimension() >= 2"], "L['x']"),
                (ASTCode,),
            ),
            (
                FakeGuard(
                    "TYPE_MATCH",
                    ["___check_type_id(L['xs'][0], 2), type=<class 'torch.Tensor'>"],
                    "L['xs'][0]",
                ),
                (TypeIdCode,),
            ),
            (
                FakeGuard("UNKNOWN", ["L['x'] >= 0"], "L['x']"),
                (ASTCode,),
            ),
            (FakeGuard("UNKNOWN", ["unsupported()"], "G['x']", GuardSource.GLOBAL), ()),
        )
        for guard, code_types in cases:
            with self.subTest(kind=guard.create_fn):
                self._assert_parsed(guard, code_types)

    def test_tensor_metadata(self) -> None:
        code_list = [
            "___check_type_id(L['xs'][0], 1), type=<class 'torch.Tensor'>",
            "str(L['xs'][0].dtype) == 'torch.float32'",
            "str(L['xs'][0].device) == 'cuda:1'",
            "L['xs'][0].requires_grad == True",
            "L['xs'][0].ndimension() == 4",
            "hasattr(L['xs'][0], '_dynamo_static_indices') == False",
        ]
        parsed = self._assert_parsed(
            FakeGuard("TENSOR_MATCH", code_list, "L['xs'][0]"),
            (
                TypeIdCode,
                TensorDTypeCode,
                TensorDeviceCode,
                RequiresGradCode,
                TensorRankCode,
                DynamoAttributeAbsentCode,
            ),
        )
        guard = FakeGuard("TENSOR_MATCH", code_list, "L['xs'][0]")
        self.assertEqual(len(parse_guards([guard, guard])), len(code_list))
        _, dtype, device, grad, rank, attribute = parsed
        assert isinstance(dtype, TensorDTypeCode)
        assert isinstance(device, TensorDeviceCode)
        assert isinstance(grad, RequiresGradCode)
        assert isinstance(rank, TensorRankCode)
        assert isinstance(attribute, DynamoAttributeAbsentCode)
        self.assertEqual(dtype.expected, "torch.float32")
        self.assertEqual(device.expected, "cuda:1")
        self.assertTrue(grad.expected)
        self.assertEqual(rank.expected, 4)
        self.assertEqual(attribute.attribute, "_dynamo_static_indices")


if __name__ == "__main__":
    unittest.main()
