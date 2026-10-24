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
from trident.guards.codes import (
    ASTCode,
    ConstantCode,
    DynamoAttributeAbsentCode,
    GuardCode,
    RequiresGradCode,
    TensorDeviceCode,
    TensorDTypeCode,
    TensorRankCode,
    TypeIdCode,
)
from trident.guards.kinds import (
    ConstantMatchGuard,
    DuplicateInputGuard,
    Guard,
    IgnoredGuard,
    SequenceLengthGuard,
    ShapeEnvGuard,
    TensorMatchGuard,
    TypeMatchGuard,
)
from trident.guards.local import Local
from trident.input import InputNodeBuilder, InputTableBuilder

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
            function = module.body.operations[0]
            block = function.regions[0].blocks[0]
            with ir.InsertionPoint(block.operations[-1]):
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                result = code.build(None, context)  # type: ignore[arg-type]
            self.assertTrue(module.operation.verify())
            result_type = result.type
            module_text = str(module)
        return result_type, module_text

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
                builder = InputTableBuilder(
                    [("xs", InputNodeBuilder(array, [InputNodeBuilder(type_)]))],
                    ["xs"],
                )
                text = "len(L['xs'][0]) == 2"
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                module = ir.Module.create()
                result_type = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
                with ir.InsertionPoint(module.body):
                    function = func.FuncOp("guard", ([array], [result_type]))
                    block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    results = code.build_result(builder.build(block.arguments), context)
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

    def test_invalid_comparisons(self) -> None:
        for text, message in (
            ("L['x'] is L['y'] is L['z']", "identity guard comparison must be binary"),
            ("1 in (1, 2)", "unsupported guard comparison operator"),
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(AssertionError, message),
            ):
                self._build_expression(text)

    def test_supported_expressions(self) -> None:
        cases = (
            ("bitwise_or", "(1 | 2) == 3", tvm_ffi_d.BoolType, ("arith.ori",)),
            (
                "boolean_logic",
                "True and not False",
                tvm_ffi_d.BoolType,
                ("arith.cmpi",),
            ),
            ("boolean_or", "True | False", tvm_ffi_d.BoolType, ("arith.ori",)),
            ("chained_comparison", "1 == 1 <= 3", tvm_ffi_d.BoolType, ("arith.cmpi",)),
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
            ("integer_addition", "1 + 2", tvm_ffi_d.IntType, ("arith.addi",)),
            (
                "mixed_boolean_float",
                "True + 2.0",
                tvm_ffi_d.FloatType,
                ("arith.extui", "arith.sitofp", "arith.addf"),
            ),
            (
                "mixed_boolean_integer",
                "True | 2",
                tvm_ffi_d.IntType,
                ("arith.extui", "arith.ori"),
            ),
            ("negation", "-1", tvm_ffi_d.IntType, ("arith.subi",)),
            ("select", "1 if True else 2", tvm_ffi_d.IntType, ("scf.if",)),
            (
                "string_equality",
                "'name' == 'name'",
                tvm_ffi_d.BoolType,
                ("tvm_ffi.eq", "tvm_ffi.to"),
            ),
            ("true_division", "5 / 2", tvm_ffi_d.FloatType, ("arith.divf",)),
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
        self.assertIn("tvm_ffi.constant.bool false", module)


class GuardParsingTest(unittest.TestCase):
    """Guard dispatch, code composition, metadata and rejected inputs."""

    def _assert_parsed(
        self,
        guard: FakeGuard,
        kind: type[Guard],
        code_types: tuple[type[GuardCode], ...],
    ) -> Guard:
        parsed = Guard.parse(guard)
        self.assertIsInstance(parsed, kind)
        assert parsed is not None
        self.assertEqual(tuple(type(item) for item in parsed.codes), code_types)
        self.assertEqual(parsed.source, Local.parse(guard.name))
        return parsed

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
                    ConstantMatchGuard,
                    (ConstantCode,),
                )
                code = parsed.codes[0]
                assert isinstance(code, ConstantCode)
                self.assertEqual(
                    ast.dump(code.expression, include_attributes=False),
                    ast.dump(
                        ast.parse(text, mode="eval").body, include_attributes=False
                    ),
                )

    def test_invalid_device(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            Guard.parse(
                FakeGuard(
                    "TENSOR_MATCH",
                    ["str(L['x'].device) == 'not-a-device'"],
                    "L['x']",
                )
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
                "constant_multiple_codes",
                FakeGuard("CONSTANT_MATCH", ["L['x'] == 1", "L['x'] == 2"], "L['x']"),
            ),
            (
                "constant_not_equal",
                FakeGuard("CONSTANT_MATCH", ["L['value'] != 1"], "L['value']"),
            ),
            (
                "constant_reversed_operands",
                FakeGuard("CONSTANT_MATCH", ["1 == L['value']"], "L['value']"),
            ),
            (
                "shape_env_local_source",
                FakeGuard("SHAPE_ENV", ["L['x'].size()[0] == 2"], "L['x']"),
            ),
            (
                "tensor_rank_inequality",
                FakeGuard("TENSOR_MATCH", ["L['x'].ndimension() >= 2"], "L['x']"),
            ),
            (
                "type_match_object_identity",
                FakeGuard("TYPE_MATCH", ["___check_obj_id(L['x'], 1)"], "L['x']"),
            ),
            ("unknown_kind", FakeGuard("UNKNOWN", None)),
        )
        for name, guard in cases:
            with self.subTest(case=name):
                self.assertIsNone(Guard.parse(guard))

    def test_supported_kinds(self) -> None:
        cases = (
            (
                FakeGuard("DUPLICATE_INPUT", ["L['x'] is L['y']"], "L['x']"),
                DuplicateInputGuard,
                (ASTCode,),
            ),
            (FakeGuard("GLOBAL_STATE", None), IgnoredGuard, ()),
            (
                FakeGuard(
                    "SEQUENCE_LENGTH",
                    [
                        "___check_type_id(L['xs'], 1), type=<class 'list'>",
                        "len(L['xs']) == 2",
                    ],
                    "L['xs']",
                ),
                SequenceLengthGuard,
                (TypeIdCode, ASTCode),
            ),
            (
                FakeGuard(
                    "SHAPE_ENV",
                    [
                        "2 <= L['x'].size()[0] <= 8",
                        "L['x'].size()[1] == 4 * L['y'].size()[1]",
                    ],
                ),
                ShapeEnvGuard,
                (ASTCode, ASTCode),
            ),
            (
                FakeGuard(
                    "TYPE_MATCH",
                    ["___check_type_id(L['xs'][0], 2), type=<class 'torch.Tensor'>"],
                    "L['xs'][0]",
                ),
                TypeMatchGuard,
                (TypeIdCode,),
            ),
        )
        for guard, kind, code_types in cases:
            with self.subTest(kind=guard.create_fn):
                self._assert_parsed(guard, kind, code_types)

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
            TensorMatchGuard,
            (
                TypeIdCode,
                TensorDTypeCode,
                TensorDeviceCode,
                RequiresGradCode,
                TensorRankCode,
                DynamoAttributeAbsentCode,
            ),
        )
        _, dtype, device, grad, rank, attribute = parsed.codes
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
