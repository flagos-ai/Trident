# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""TVM FFI Python type and operation binding contracts."""

import unittest

from trident.core import ir
from trident.core.dialects import func, tvm_ffi
from trident.utils.monad import pure
from trident.utils.types import value_type
from trident.utils.values import constant

from test.support.ir import new_context


class BindingsTest(unittest.TestCase):
    def test_abi_type_spellings(self) -> None:
        context = new_context()
        with context:
            for name, type_class in (
                ("any", tvm_ffi.api.AnyType),
                ("bool", tvm_ffi.abi.BoolType),
                ("abi.any", tvm_ffi.abi.AnyType),
                ("abi.array", tvm_ffi.abi.ArrayType),
                ("abi.exception", tvm_ffi.abi.ExceptionType),
                ("abi.function", tvm_ffi.abi.FunctionType),
                ("abi.str", tvm_ffi.abi.StrType),
                ("abi.tensor", tvm_ffi.abi.TensorType),
                ("dtype", tvm_ffi.abi.DTypeType),
                ("device", tvm_ffi.abi.DeviceType),
                ("float", tvm_ffi.abi.FloatType),
                ("function", tvm_ffi.api.FunctionType),
                ("int", tvm_ffi.abi.IntType),
                ("none", tvm_ffi.abi.NoneType),
                ("object", tvm_ffi.abi.ObjectType),
                ("pack.arguments", tvm_ffi.abi.PackArgumentsType),
                ("raw_str", tvm_ffi.abi.RawStrType),
                ("small_str", tvm_ffi.abi.SmallStrType),
                ("str", tvm_ffi.api.StrType),
                ("tensor", tvm_ffi.api.TensorType),
            ):
                with self.subTest(name=name):
                    type_ = type_class.get()
                    dialect = (
                        "api"
                        if type_class.__module__.endswith("_tvm_ffi_api")
                        else "abi"
                    )
                    mnemonic = name.removeprefix("abi.")
                    spelling = f"!tvm_ffi_{dialect}.{mnemonic}"
                    self.assertEqual(f"{type_}", spelling)
                    parsed = ir.Type.parse(spelling)
                    self.assertIsInstance(parsed, type_class)
                    self.assertEqual(parsed, type_)
            self.assertNotEqual(tvm_ffi.api.AnyType.get(), tvm_ffi.abi.AnyType.get())
            self.assertNotIsInstance(
                ir.Type.parse("!tvm_ffi_abi.any"), tvm_ffi.api.AnyType
            )
            exception = tvm_ffi.api.ExceptionType.get()
            self.assertEqual(f"{exception}", "!tvm_ffi_api.exception")
            self.assertIsInstance(
                ir.Type.parse("!tvm_ffi_api.exception"), tvm_ffi.api.ExceptionType
            )
            union = tvm_ffi.api.UnionType.get([tvm_ffi.abi.IntType.get(), exception])
            self.assertEqual(
                f"{union}",
                "!tvm_ffi_api.union<!tvm_ffi_abi.int, !tvm_ffi_api.exception>",
            )
            parsed_union = ir.Type.parse(f"{union}")
            self.assertIsInstance(parsed_union, tvm_ffi.api.UnionType)
            self.assertEqual(parsed_union.types, union.types)
            for member in (
                ir.IntegerType.get_signless(64),
                tvm_ffi.abi.ObjectType.get(),
                tvm_ffi.abi.PackArgumentsType.get(),
                tvm_ffi.monad.ResultType.get(tvm_ffi.abi.IntType.get()),
            ):
                with (
                    self.subTest(member=f"{member}"),
                    self.assertRaisesRegex(
                        ValueError, "union members must be TVM FFI C ABI types"
                    ),
                ):
                    tvm_ffi.api.UnionType.get([tvm_ffi.abi.IntType.get(), member])
            result = tvm_ffi.monad.ResultType.get(union)
            self.assertEqual(f"{result}", f"!tvm_ffi_monad.result<{union}>")
            parsed_result = ir.Type.parse(f"{result}")
            self.assertIsInstance(parsed_result, tvm_ffi.monad.ResultType)
            self.assertEqual(parsed_result.payload_type, union)

    def test_independent_dialect_registration(self) -> None:
        with ir.Context() as context:
            tvm_ffi.abi.register_dialect(context)
            integer = ir.Type.parse("!tvm_ffi_abi.int")
            self.assertIsInstance(integer, tvm_ffi.abi.IntType)
            self.assertIsInstance(
                ir.Type.parse("!tvm_ffi_abi.tensor"), tvm_ffi.abi.TensorType
            )
            with self.assertRaises(IndexError):
                _ = context.dialects["tvm_ffi_api"]
        with ir.Context() as context:
            tvm_ffi.api.register_dialect(context)
            tensor = ir.Type.parse("!tvm_ffi_api.tensor")
            self.assertIsInstance(tensor, tvm_ffi.api.TensorType)
            self.assertIsInstance(
                ir.Type.parse("!tvm_ffi_abi.int"), tvm_ffi.abi.IntType
            )
            self.assertNotEqual(tensor, tvm_ffi.abi.TensorType.get())
            with self.assertRaises(IndexError):
                _ = context.dialects["tvm_ffi_monad"]

    def test_arith_and_monad_registration(self) -> None:
        with ir.Context() as context, ir.Location.unknown():
            tvm_ffi.arith.register_dialect(context)
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                value = constant(2)
                added = tvm_ffi.arith.add(value, value)
            self.assertEqual(added.owner.name, "tvm_ffi_arith.add")
            self.assertIsInstance(added.type, tvm_ffi.abi.IntType)
            self.assertTrue(module.operation.verify())
            for name in ("tvm_ffi_api", "tvm_ffi_monad"):
                with self.subTest(dialect=name), self.assertRaises(IndexError):
                    _ = context.dialects[name]
        with ir.Context() as context, ir.Location.unknown():
            tvm_ffi.monad.register_dialect(context)
            payload = tvm_ffi.api.TensorType.get()
            result = tvm_ffi.monad.ResultType.get(payload)
            parsed = ir.Type.parse(f"{result}")
            self.assertIsInstance(parsed, tvm_ffi.monad.ResultType)
            self.assertEqual(parsed.payload_type, payload)
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                successful = tvm_ffi.monad.pure(
                    tvm_ffi.monad.ResultType.get(tvm_ffi.abi.IntType.get()),
                    constant(2),
                )
                success, value = tvm_ffi.monad.unpack(successful)
            self.assertEqual(success.type, ir.IntegerType.get_signless(1))
            self.assertIsInstance(value.type, tvm_ffi.api.AnyType)
            self.assertTrue(module.operation.verify())
        self.assertFalse(hasattr(tvm_ffi.api, "ResultType"))
        self.assertFalse(hasattr(tvm_ffi.api, "arith_add"))
        self.assertFalse(hasattr(tvm_ffi.api, "monad_bind"))

    def test_bind_get_entry_block(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            payload_types = [tvm_ffi.abi.IntType.get(), tvm_ffi.abi.NoneType.get()]
            result_types = [
                tvm_ffi.monad.ResultType.get(type_) for type_ in payload_types
            ]
            locations = [ir.Location.name("integer"), ir.Location.name("none")]
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", (result_types, []))
                entry = function.add_entry_block()
            with ir.InsertionPoint(entry):
                operation = tvm_ffi.monad.BindOp([], entry.arguments)
                self.assertIsNone(operation.get_entry_block())
                self.assertEqual(len(operation.body.blocks), 0)
                body = operation.add_entry_block(arg_locs=locations)
                self.assertEqual(ir.InsertionPoint.current.block, entry)
                self.assertEqual(
                    [value.type for value in body.arguments], payload_types
                )
                self.assertEqual(
                    [value.location for value in body.arguments], locations
                )
                self.assertEqual(list(operation.arguments), list(body.arguments))
                self.assertEqual(operation.get_entry_block(), body)
                self.assertEqual(operation.add_entry_block(), body)
                self.assertEqual(len(operation.body.blocks), 1)
                self.assertEqual(ir.InsertionPoint.current.block, entry)
                with ir.InsertionPoint(body):
                    tvm_ffi.monad.yield_([])

                empty = tvm_ffi.monad.BindOp([], [])
                self.assertIsNone(empty.get_entry_block())
                with ir.InsertionPoint(empty.add_entry_block()):
                    self.assertEqual(len(empty.arguments), 0)
                    tvm_ffi.monad.yield_([])

                [empty_block] = empty.body.blocks
                self.assertEqual(empty.get_entry_block(), empty_block)
                self.assertEqual(empty.add_entry_block(), empty_block)
                self.assertEqual(len(empty.body.blocks), 1)
                func.return_([])

            self.assertTrue(module.operation.verify())

    def test_container_types(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            integer = tvm_ffi.abi.IntType.get()
            for mapping in ({}, {"key": [1, "value"]}):
                with self.subTest(mapping=mapping):
                    self.assertEqual(
                        value_type(mapping),
                        tvm_ffi.api.ArrayType.get(tvm_ffi.api.AnyType.get()),
                    )
            array = tvm_ffi.api.ArrayType.get(integer)
            tuple_ = tvm_ffi.api.TupleType.get([array, integer, integer])
            for value, cls in (
                (array, tvm_ffi.api.ArrayType),
                (tuple_, tvm_ffi.api.TupleType),
            ):
                self.assertIsInstance(ir.Type.parse(f"{value}"), cls)
                self.assertEqual(ir.Type.parse(f"{value}"), value)
            self.assertEqual(array.element_type, integer)
            self.assertEqual(tuple_.types, (array, integer, integer))
            abi_members = (
                tvm_ffi.api.AnyType.get(),
                array,
                tvm_ffi.api.ExceptionType.get(),
                tvm_ffi.api.FunctionType.get(),
                tvm_ffi.abi.NoneType.get(),
                tuple_,
                tvm_ffi.api.UnionType.get([integer, tvm_ffi.abi.NoneType.get()]),
            )
            for element in abi_members:
                with self.subTest(array_element=element):
                    abi_array = tvm_ffi.api.ArrayType.get(element)
                    self.assertEqual(abi_array.element_type, element)
                    self.assertEqual(ir.Type.parse(f"{abi_array}"), abi_array)
            abi_tuple = tvm_ffi.api.TupleType.get(abi_members)
            self.assertEqual(abi_tuple.types, abi_members)
            self.assertEqual(ir.Type.parse(f"{abi_tuple}"), abi_tuple)
            empty = tvm_ffi.api.TupleType.get([])
            self.assertEqual(empty.types, ())
            self.assertEqual(ir.Type.parse(f"{empty}"), empty)
            for type_ in (
                ir.IntegerType.get_signless(64),
                ir.RankedTensorType.get([2], ir.IntegerType.get_signless(64)),
                tvm_ffi.abi.ObjectType.get(),
                tvm_ffi.abi.PackArgumentsType.get(),
                tvm_ffi.monad.ResultType.get(integer),
            ):
                with self.assertRaisesRegex(
                    ValueError, "array element must be a TVM FFI C ABI type"
                ):
                    tvm_ffi.api.ArrayType.get(type_)
                with self.assertRaisesRegex(
                    ValueError, "tuple elements must be TVM FFI C ABI types"
                ):
                    tvm_ffi.api.TupleType.get([type_])
            other = new_context()
            with other:
                foreign = tvm_ffi.abi.IntType.get()
            with self.assertRaises(ValueError):
                tvm_ffi.api.ArrayType.get(foreign)
            with self.assertRaises(ValueError):
                tvm_ffi.api.TupleType.get([foreign])

    def test_function_entry_blocks(self) -> None:
        for name, operation_class, argument_class in (
            ("packed", tvm_ffi.api.PackFuncOp, tvm_ffi.abi.PackArgumentsType),
            ("typed", tvm_ffi.api.FuncOp, tvm_ffi.abi.IntType),
        ):
            with self.subTest(function=name), new_context(), ir.Location.unknown():
                module = ir.Module.create()
                argument_type = argument_class.get()
                any_type = tvm_ffi.api.AnyType.get()
                location = ir.Location.name("arguments")
                with ir.InsertionPoint(module.body):
                    operation = operation_class(
                        name,
                        ir.TypeAttr.get(
                            ir.FunctionType.get([argument_type], [any_type])
                        ),
                    )
                    self.assertIsNone(operation.get_entry_block())
                    body = operation.add_entry_block(arg_locs=[location])
                    self.assertEqual(ir.InsertionPoint.current.block, module.body)
                    self.assertEqual(
                        [value.type for value in body.arguments], [argument_type]
                    )
                    [argument] = body.arguments
                    self.assertEqual(argument.location, location)
                    self.assertEqual(operation.get_entry_block(), body)
                    self.assertEqual(operation.add_entry_block(), body)
                    self.assertEqual(len(operation.body.blocks), 1)
                with ir.InsertionPoint(body):
                    success = tvm_ffi.abi.constant_bool(True)
                    payload = tvm_ffi.api.upcast(any_type, tvm_ffi.abi.constant_none())
                    tvm_ffi.api.return_(success, payload)
                self.assertTrue(module.operation.verify())

    def test_local_call_bindings(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            payload = tvm_ffi.api.TensorType.get()
            result = tvm_ffi.monad.ResultType.get(payload)
            success = tvm_ffi.abi.BoolType.get()
            value = tvm_ffi.api.AnyType.get()
            with ir.InsertionPoint(module.body):
                tvm_ffi.api.func(
                    "callee",
                    ir.TypeAttr.get(ir.FunctionType.get([payload], [payload])),
                )
                caller = func.FuncOp("caller", ([payload], [result, success, value]))
                with ir.InsertionPoint(caller.add_entry_block()):
                    arguments = list(caller.arguments)
                    monad_call = tvm_ffi.monad.CallOp(result, "callee", arguments)
                    call = tvm_ffi.api.CallOp("callee", arguments)
                    self.assertIsInstance(
                        monad_call.result.type, tvm_ffi.monad.ResultType
                    )
                    self.assertEqual(monad_call.result.type.payload_type, payload)
                    self.assertIsInstance(call.success.type, tvm_ffi.abi.BoolType)
                    self.assertIsInstance(call.value.type, tvm_ffi.api.AnyType)
                    func.return_([monad_call.result, call.success, call.value])
            self.assertTrue(module.operation.verify())
            parsed = ir.Module.parse(f"{module}")
            _, caller = parsed.body.operations
            [region] = caller.regions
            [block] = region.blocks
            monad_call, call, _ = block.operations
            self.assertIsInstance(monad_call, tvm_ffi.monad.CallOp)
            self.assertIsInstance(call, tvm_ffi.api.CallOp)
            self.assertIsInstance(call.success.type, tvm_ffi.abi.BoolType)
            self.assertIsInstance(call.value.type, tvm_ffi.api.AnyType)

    def test_monad_unpack(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                result = pure(constant(7))
                unpacked = tvm_ffi.monad.UnpackOp(result)
                self.assertEqual(unpacked.success.type, ir.IntegerType.get_signless(1))
                self.assertEqual(unpacked.value.type, tvm_ffi.api.AnyType.get())
                self.assertTrue(module.operation.verify())

    def test_result_payload_types(self) -> None:
        with self.assertRaises(TypeError):
            tvm_ffi.monad.ResultType.get()
        with self.assertRaises(TypeError):
            tvm_ffi.monad.ResultType.get(None)
        with new_context(), ir.Location.unknown():
            integer = tvm_ffi.abi.IntType.get()
            for payload in (
                tvm_ffi.api.AnyType.get(),
                tvm_ffi.api.ArrayType.get(integer),
                tvm_ffi.abi.ArrayType.get(),
                tvm_ffi.api.FunctionType.get(),
                integer,
                tvm_ffi.abi.NoneType.get(),
                tvm_ffi.api.TupleType.get([integer]),
                tvm_ffi.api.UnionType.get([integer, tvm_ffi.abi.NoneType.get()]),
            ):
                with self.subTest(payload=payload):
                    result = tvm_ffi.monad.ResultType.get(payload)
                    self.assertEqual(result.payload_type, payload)
                    self.assertEqual(ir.Type.parse(f"{result}"), result)
            for payload in (
                ir.IntegerType.get_signless(64),
                tvm_ffi.abi.ObjectType.get(),
                tvm_ffi.abi.PackArgumentsType.get(),
                tvm_ffi.monad.ResultType.get(integer),
            ):
                with (
                    self.subTest(payload=payload),
                    self.assertRaisesRegex(
                        ValueError, "Result payload must be a TVM FFI C ABI type"
                    ),
                ):
                    tvm_ffi.monad.ResultType.get(payload)
            with new_context():
                foreign = tvm_ffi.abi.IntType.get()
            with self.assertRaisesRegex(ValueError, "construction context"):
                tvm_ffi.monad.ResultType.get(foreign)
