# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""TVM FFI Python type and operation binding contracts."""

import unittest

from trident.core import ir
from trident.core.dialects import func
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import constant, pure, value_type

from test.support.ir import new_context


class BindingsTest(unittest.TestCase):
    def test_abi_type_spellings(self) -> None:
        context = new_context()
        with context:
            for name, type_class in (
                ("any", tvm_ffi_d.AnyType),
                ("bool", tvm_ffi_d.BoolType),
                ("c.any", tvm_ffi_d.CAnyType),
                ("c.array", tvm_ffi_d.CArrayType),
                ("c.exception", tvm_ffi_d.CExceptionType),
                ("c.function", tvm_ffi_d.CFunctionType),
                ("c.str", tvm_ffi_d.CStrType),
                ("c.tensor", tvm_ffi_d.CTensorType),
                ("dtype", tvm_ffi_d.DTypeType),
                ("device", tvm_ffi_d.DeviceType),
                ("float", tvm_ffi_d.FloatType),
                ("function", tvm_ffi_d.FunctionType),
                ("int", tvm_ffi_d.IntType),
                ("none", tvm_ffi_d.NoneType),
                ("object", tvm_ffi_d.ObjectType),
                ("pack.arguments", tvm_ffi_d.PackArgumentsType),
                ("raw_str", tvm_ffi_d.RawStrType),
                ("small_str", tvm_ffi_d.SmallStrType),
                ("str", tvm_ffi_d.StrType),
                ("tensor", tvm_ffi_d.TensorType),
            ):
                with self.subTest(name=name):
                    type_ = type_class.get()
                    self.assertEqual(f"{type_}", f"!tvm_ffi.{name}")
                    parsed = ir.Type.parse(f"!tvm_ffi.{name}")
                    self.assertIsInstance(parsed, type_class)
                    self.assertEqual(parsed, type_)
            self.assertNotEqual(tvm_ffi_d.AnyType.get(), tvm_ffi_d.CAnyType.get())
            self.assertNotIsInstance(ir.Type.parse("!tvm_ffi.c.any"), tvm_ffi_d.AnyType)
            exception = tvm_ffi_d.ExceptionType.get()
            self.assertEqual(f"{exception}", "!tvm_ffi.exception")
            self.assertIsInstance(
                ir.Type.parse("!tvm_ffi.exception"), tvm_ffi_d.ExceptionType
            )
            union = tvm_ffi_d.UnionType.get([tvm_ffi_d.IntType.get(), exception])
            self.assertEqual(
                f"{union}", "!tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>"
            )
            parsed_union = ir.Type.parse(f"{union}")
            self.assertIsInstance(parsed_union, tvm_ffi_d.UnionType)
            self.assertEqual(parsed_union.types, union.types)
            for member in (
                ir.IntegerType.get_signless(64),
                tvm_ffi_d.ObjectType.get(),
                tvm_ffi_d.PackArgumentsType.get(),
                tvm_ffi_d.ResultType.get(tvm_ffi_d.IntType.get()),
            ):
                with (
                    self.subTest(member=f"{member}"),
                    self.assertRaisesRegex(
                        ValueError, "union members must be TVM FFI C ABI types"
                    ),
                ):
                    tvm_ffi_d.UnionType.get([tvm_ffi_d.IntType.get(), member])
            result = tvm_ffi_d.ResultType.get(union)
            self.assertEqual(f"{result}", f"!tvm_ffi.monad.result<{union}>")
            parsed_result = ir.Type.parse(f"{result}")
            self.assertIsInstance(parsed_result, tvm_ffi_d.ResultType)
            self.assertEqual(parsed_result.payload_type, union)

    def test_bind_get_entry_block(self) -> None:
        with new_context(), ir.Location.unknown():
            module = ir.Module.create()
            payload_types = [tvm_ffi_d.IntType.get(), tvm_ffi_d.NoneType.get()]
            result_types = [tvm_ffi_d.ResultType.get(type_) for type_ in payload_types]
            locations = [ir.Location.name("integer"), ir.Location.name("none")]
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("test", (result_types, []))
                entry = function.add_entry_block()
            with ir.InsertionPoint(entry):
                operation = tvm_ffi_d.MonadBindOp([], entry.arguments)
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
                    tvm_ffi_d.monad_yield([])

                empty = tvm_ffi_d.MonadBindOp([], [])
                self.assertIsNone(empty.get_entry_block())
                with ir.InsertionPoint(empty.add_entry_block()):
                    self.assertEqual(len(empty.arguments), 0)
                    tvm_ffi_d.monad_yield([])

                self.assertEqual(empty.get_entry_block(), empty.body.blocks[0])
                self.assertEqual(empty.add_entry_block(), empty.body.blocks[0])
                self.assertEqual(len(empty.body.blocks), 1)
                func.return_([])

            self.assertTrue(module.operation.verify())

    def test_container_types(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            integer = tvm_ffi_d.IntType.get()
            for mapping in ({}, {"key": [1, "value"]}):
                with self.subTest(mapping=mapping):
                    self.assertEqual(
                        value_type(mapping),
                        tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get()),
                    )
            array = tvm_ffi_d.ArrayType.get(integer)
            tuple_ = tvm_ffi_d.TupleType.get([array, integer, integer])
            for value, cls in (
                (array, tvm_ffi_d.ArrayType),
                (tuple_, tvm_ffi_d.TupleType),
            ):
                self.assertIsInstance(ir.Type.parse(f"{value}"), cls)
                self.assertEqual(ir.Type.parse(f"{value}"), value)
            self.assertEqual(array.element_type, integer)
            self.assertEqual(tuple_.types, (array, integer, integer))
            abi_members = (
                tvm_ffi_d.AnyType.get(),
                array,
                tvm_ffi_d.ExceptionType.get(),
                tvm_ffi_d.FunctionType.get(),
                tvm_ffi_d.NoneType.get(),
                tuple_,
                tvm_ffi_d.UnionType.get([integer, tvm_ffi_d.NoneType.get()]),
            )
            for element in abi_members:
                with self.subTest(array_element=element):
                    abi_array = tvm_ffi_d.ArrayType.get(element)
                    self.assertEqual(abi_array.element_type, element)
                    self.assertEqual(ir.Type.parse(f"{abi_array}"), abi_array)
            abi_tuple = tvm_ffi_d.TupleType.get(abi_members)
            self.assertEqual(abi_tuple.types, abi_members)
            self.assertEqual(ir.Type.parse(f"{abi_tuple}"), abi_tuple)
            empty = tvm_ffi_d.TupleType.get([])
            self.assertEqual(empty.types, ())
            self.assertEqual(ir.Type.parse(f"{empty}"), empty)
            for type_ in (
                ir.IntegerType.get_signless(64),
                ir.RankedTensorType.get([2], ir.IntegerType.get_signless(64)),
                tvm_ffi_d.ObjectType.get(),
                tvm_ffi_d.PackArgumentsType.get(),
                tvm_ffi_d.ResultType.get(integer),
            ):
                with self.assertRaisesRegex(
                    ValueError, "array element must be a TVM FFI C ABI type"
                ):
                    tvm_ffi_d.ArrayType.get(type_)
                with self.assertRaisesRegex(
                    ValueError, "tuple elements must be TVM FFI C ABI types"
                ):
                    tvm_ffi_d.TupleType.get([type_])
            other = new_context()
            with other:
                foreign = tvm_ffi_d.IntType.get()
            with self.assertRaises(ValueError):
                tvm_ffi_d.ArrayType.get(foreign)
            with self.assertRaises(ValueError):
                tvm_ffi_d.TupleType.get([foreign])

    def test_function_entry_blocks(self) -> None:
        for name, operation_class, argument_class in (
            ("packed", tvm_ffi_d.PackFuncOp, tvm_ffi_d.PackArgumentsType),
            ("typed", tvm_ffi_d.FuncOp, tvm_ffi_d.IntType),
        ):
            with self.subTest(function=name), new_context(), ir.Location.unknown():
                module = ir.Module.create()
                argument_type = argument_class.get()
                any_type = tvm_ffi_d.AnyType.get()
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
                    self.assertEqual(body.arguments[0].location, location)
                    self.assertEqual(operation.get_entry_block(), body)
                    self.assertEqual(operation.add_entry_block(), body)
                    self.assertEqual(len(operation.body.blocks), 1)
                with ir.InsertionPoint(body):
                    success = tvm_ffi_d.constant_bool(True)
                    payload = tvm_ffi_d.upcast(any_type, tvm_ffi_d.constant_none())
                    tvm_ffi_d.return_(success, payload)
                self.assertTrue(module.operation.verify())

    def test_local_call_bindings(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            payload = tvm_ffi_d.TensorType.get()
            result = tvm_ffi_d.ResultType.get(payload)
            success = tvm_ffi_d.BoolType.get()
            value = tvm_ffi_d.AnyType.get()
            with ir.InsertionPoint(module.body):
                tvm_ffi_d.func(
                    "callee",
                    ir.TypeAttr.get(ir.FunctionType.get([payload], [payload])),
                )
                caller = func.FuncOp("caller", ([payload], [result, success, value]))
                with ir.InsertionPoint(caller.add_entry_block()):
                    arguments = list(caller.arguments)
                    monad_call = tvm_ffi_d.MonadCallOp(result, "callee", arguments)
                    call = tvm_ffi_d.CallOp("callee", arguments)
                    self.assertIsInstance(monad_call.result.type, tvm_ffi_d.ResultType)
                    self.assertEqual(monad_call.result.type.payload_type, payload)
                    self.assertIsInstance(call.success.type, tvm_ffi_d.BoolType)
                    self.assertIsInstance(call.value.type, tvm_ffi_d.AnyType)
                    func.return_([monad_call.result, call.success, call.value])
            self.assertTrue(module.operation.verify())
            parsed = ir.Module.parse(f"{module}")
            calls = list(parsed.body.operations[1].regions[0].blocks[0].operations)
            self.assertIsInstance(calls[0], tvm_ffi_d.MonadCallOp)
            self.assertIsInstance(calls[1], tvm_ffi_d.CallOp)
            self.assertIsInstance(calls[1].success.type, tvm_ffi_d.BoolType)
            self.assertIsInstance(calls[1].value.type, tvm_ffi_d.AnyType)

    def test_monad_unpack(self) -> None:
        context = new_context()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                [result] = pure([constant(7)])
                unpacked = tvm_ffi_d.MonadUnpackOp(result)
                self.assertEqual(unpacked.success.type, ir.IntegerType.get_signless(1))
                self.assertEqual(unpacked.value.type, tvm_ffi_d.AnyType.get())
                self.assertTrue(module.operation.verify())

    def test_result_payload_types(self) -> None:
        with self.assertRaises(TypeError):
            tvm_ffi_d.ResultType.get()
        with self.assertRaises(TypeError):
            tvm_ffi_d.ResultType.get(None)
        with new_context(), ir.Location.unknown():
            integer = tvm_ffi_d.IntType.get()
            for payload in (
                tvm_ffi_d.AnyType.get(),
                tvm_ffi_d.ArrayType.get(integer),
                tvm_ffi_d.CArrayType.get(),
                tvm_ffi_d.FunctionType.get(),
                integer,
                tvm_ffi_d.NoneType.get(),
                tvm_ffi_d.TupleType.get([integer]),
                tvm_ffi_d.UnionType.get([integer, tvm_ffi_d.NoneType.get()]),
            ):
                with self.subTest(payload=payload):
                    result = tvm_ffi_d.ResultType.get(payload)
                    self.assertEqual(result.payload_type, payload)
                    self.assertEqual(ir.Type.parse(f"{result}"), result)
            for payload in (
                ir.IntegerType.get_signless(64),
                tvm_ffi_d.ObjectType.get(),
                tvm_ffi_d.PackArgumentsType.get(),
                tvm_ffi_d.ResultType.get(integer),
            ):
                with (
                    self.subTest(payload=payload),
                    self.assertRaisesRegex(
                        ValueError, "Result payload must be a TVM FFI C ABI type"
                    ),
                ):
                    tvm_ffi_d.ResultType.get(payload)
            with new_context():
                foreign = tvm_ffi_d.IntType.get()
            with self.assertRaisesRegex(ValueError, "construction context"):
                tvm_ffi_d.ResultType.get(foreign)
