# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Shared IR builders for graph import, input handling and Dynamo guards.

Scalars stay native while computing and are boxed at FFI boundaries.
Checked calls validate lookup status, function handles and call status.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import arith, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ownership import is_value_in_scope


def _is_native_scalar(type_: ir.Type) -> bool:
    return isinstance(type_, ir.F64Type) or (
        isinstance(type_, ir.IntegerType)
        and type_.is_signless
        and type_.width in (1, 64)
    )


def _is_numeric_scalar(type_: ir.Type) -> bool:
    return _is_native_scalar(type_) or isinstance(
        type_, (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
    )


def _is_signless_integer(type_: ir.Type, width: int) -> bool:
    return (
        isinstance(type_, ir.IntegerType) and type_.is_signless and type_.width == width
    )


def _wrap_floating(
    builder: Callable[[ir.Value, ir.Value], ir.Value],
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def wrapped(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        values: list[ir.Value] = []
        for value in (lhs, rhs):
            if _is_signless_integer(value.type, 1):
                value = arith.extui(ir.IntegerType.get_signless(64), value)
            values.append(
                value
                if isinstance(value.type, ir.F64Type)
                else arith.sitofp(ir.F64Type.get(), value)
            )
        return builder(*values)

    return wrapped


def _wrap_integer(
    builder: Callable[[ir.Value, ir.Value], ir.Value],
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def wrapped(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if any(isinstance(value.type, ir.F64Type) for value in (lhs, rhs)):
            raise NotImplementedError("floating-point input to integer operation")
        else:
            return builder(
                *(
                    arith.extui(ir.IntegerType.get_signless(64), value)
                    if _is_signless_integer(value.type, 1)
                    else value
                    for value in (lhs, rhs)
                )
            )

    return wrapped


# Entries contain boolean, integer and floating-point implementations, in that order.
_BINARY_ARITHMETIC: dict[
    Callable[..., Any],
    tuple[
        Callable[[ir.Value, ir.Value], ir.Value] | None,
        Callable[[ir.Value, ir.Value], ir.Value] | None,
        Callable[[ir.Value, ir.Value], ir.Value] | None,
    ],
] = {
    max: (None, _wrap_integer(arith.maxsi), _wrap_floating(arith.maximumf)),
    min: (None, _wrap_integer(arith.minsi), _wrap_floating(arith.minimumf)),
    operator.add: (None, _wrap_integer(arith.addi), _wrap_floating(arith.addf)),
    operator.and_: (arith.andi, _wrap_integer(arith.andi), None),
    operator.eq: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.eq)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OEQ)),
    ),
    operator.floordiv: (None, _wrap_integer(arith.floordivsi), None),
    operator.ge: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sge)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OGE)),
    ),
    operator.gt: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sgt)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OGT)),
    ),
    operator.le: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sle)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OLE)),
    ),
    operator.lshift: (None, _wrap_integer(arith.shli), None),
    operator.lt: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.slt)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OLT)),
    ),
    operator.mod: (
        None,
        _wrap_integer(
            lambda lhs, rhs: arith.subi(
                lhs, arith.muli(arith.floordivsi(lhs, rhs), rhs)
            )
        ),
        None,
    ),
    operator.mul: (None, _wrap_integer(arith.muli), _wrap_floating(arith.mulf)),
    operator.ne: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.ne)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.UNE)),
    ),
    operator.or_: (arith.ori, _wrap_integer(arith.ori), None),
    operator.rshift: (None, _wrap_integer(arith.shrsi), None),
    operator.sub: (None, _wrap_integer(arith.subi), _wrap_floating(arith.subf)),
    operator.truediv: (None, None, _wrap_floating(arith.divf)),
    operator.xor: (arith.xori, _wrap_integer(arith.xori), None),
}


def array_type(types: Sequence[ir.Type]) -> tvm_ffi_d.ArrayType:
    elements = [boxed_type(type_) for type_ in types]
    element = (
        elements[0]
        if elements and all(type_ == elements[0] for type_ in elements)
        else tvm_ffi_d.AnyType.get()
    )
    return tvm_ffi_d.ArrayType.get(element)


def binary(operation: Any, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    lhs, rhs = native(lhs), native(rhs)
    boolean, integer, floating = _BINARY_ARITHMETIC[operation]
    if operation is operator.truediv or any(
        isinstance(value.type, ir.F64Type) for value in (lhs, rhs)
    ):
        builder = floating
    elif boolean is not None and all(
        _is_signless_integer(value.type, 1) for value in (lhs, rhs)
    ):
        builder = boolean
    else:
        builder = integer
    if builder is None:
        raise NotImplementedError(f"unsupported scalar operation: {operation}")
    else:
        return builder(lhs, rhs)


@contextmanager
def bind(
    inputs: Sequence[ir.Value], *, result_types: Sequence[ir.Type]
) -> Iterator[tvm_ffi_d.MonadBindOp]:
    """Expose one payload per Result inside its successful continuation."""
    assert all(isinstance(value.type, tvm_ffi_d.ResultType) for value in inputs), (
        "bind inputs must be Results"
    )
    assert all(isinstance(type_, tvm_ffi_d.ResultType) for type_ in result_types)
    operation = tvm_ffi_d.MonadBindOp(result_types, inputs)

    with ir.InsertionPoint(operation.add_entry_block()):
        yield operation


def box(value: ir.Value) -> ir.Value:
    if _is_native_scalar(value.type):
        return tvm_ffi_d.to(value)
    if isinstance(value.type, (tvm_ffi_d.RawStrType, tvm_ffi_d.SmallStrType)):
        return tvm_ffi_d.upcast(tvm_ffi_d.StrType.get(), value)
    return value


def boxed_type(type_: ir.Type) -> ir.Type:
    if _is_signless_integer(type_, 1):
        return tvm_ffi_d.BoolType.get()
    elif _is_signless_integer(type_, 64):
        return tvm_ffi_d.IntType.get()
    elif isinstance(type_, ir.F64Type):
        return tvm_ffi_d.FloatType.get()
    elif isinstance(type_, (tvm_ffi_d.RawStrType, tvm_ffi_d.SmallStrType)):
        return tvm_ffi_d.StrType.get()
    return type_


def checked_call(
    name: str, arguments: Sequence[ir.Value], result: ir.Type
) -> list[ir.Value]:
    """Return an asserting invocation wrapped as a successful Result."""
    assert all(is_value_in_scope(value) for value in arguments), (
        "call argument payload is outside its defining block"
    )
    handle = tvm_ffi_d.FunctionGetGlobalOp(name).result
    call = tvm_ffi_d.FunctionCallOp(
        boxed_type(result), handle, [box(value) for value in arguments]
    )

    return pure([call.result])


def checked_result(value: ir.Value, success: ir.Value) -> ir.Value:
    """Wrap a borrowed fallible payload, retaining it only on success."""
    type_ = make_result_types([value.type])[0]
    branch = scf.IfOp(success, [type_], has_else=True)

    with ir.InsertionPoint(branch.then_block):
        scf.YieldOp(pure([value]))
    with ir.InsertionPoint(branch.else_block):
        scf.YieldOp(tvm_ffi_d.MonadFailOp([type_]).outputs)
    (result,) = branch.results
    return result


def constant(value: Any) -> ir.Value:
    if isinstance(value, (bool, int, float)):
        return native_constant(value)
    elif value is None:
        return tvm_ffi_d.constant_none()
    elif isinstance(value, str):
        return tvm_ffi_d.constant_str(value)
    elif isinstance(value, torch.device):
        return tvm_ffi_d.constant_device(f"{value}")
    elif isinstance(value, torch.dtype):
        dtype = runtime.convert(value)
        i64 = ir.IntegerType.get_signless(64)
        return tvm_ffi_d.constant_dtype(
            [
                ir.IntegerAttr.get(i64, field)
                for field in (dtype.type_code, dtype.bits, dtype.lanes)
            ]
        )
    elif isinstance(value, torch.Tensor):
        # Lifted literals reuse the existing owned runtime materialization.
        cpu = value.detach().cpu().contiguous()
        if cpu.dtype == torch.bfloat16:
            elements = ir.DenseElementsAttr.get(
                cpu.view(torch.int16).numpy(),
                type=ir.RankedTensorType.get(cpu.shape, ir.BF16Type.get()),
            )
        else:
            elements = ir.DenseElementsAttr.get(cpu.numpy())
        return tvm_ffi_d.TensorLiteralOp(elements).result
    else:
        raise NotImplementedError(f"unsupported FX constant: {value!r}")


def constant_type(value: Any) -> ir.Type:
    """Type produced by a leaf constant, before boxing at an FFI boundary."""
    return (
        tvm_ffi_d.StrType.get()
        if isinstance(value, str)
        else native_type(value_type(value))
    )


def container_create(values: Sequence[ir.Value], type_: ir.Type) -> list[ir.Value]:
    elements = [box(value) for value in values]
    types = (
        type_.types
        if isinstance(type_, tvm_ffi_d.TupleType)
        else [type_.element_type] * len(elements)
    )
    elements = [
        tvm_ffi_d.upcast(target, value)
        for value, target in zip(elements, types, strict=True)
    ]
    if isinstance(type_, tvm_ffi_d.TupleType):
        value = tvm_ffi_d.TupleConstructOp(type_, elements).result
    else:
        value = tvm_ffi_d.ArrayCreateOp(type_, elements).result
    return pure([value])


def container_get(
    container: ir.Value, index: int | ir.Value, target: ir.Type
) -> list[ir.Value]:
    type_ = container.type
    if isinstance(type_, tvm_ffi_d.TupleType):
        assert isinstance(index, int)
        assert -len(type_.types) <= index < len(type_.types), "tuple index out of range"
        payload = type_.types[index]
        value = tvm_ffi_d.TupleUnconstructOp(type_.types, container).elements[index]
    elif isinstance(type_, tvm_ffi_d.ArrayType):
        payload = type_.element_type
        value = tvm_ffi_d.ArrayGetOp(
            container, native_constant(index) if isinstance(index, int) else index
        ).result
    else:
        return checked_call(
            "ffi.ArrayGetItem", [container, native_constant(index)], target
        )
    [result] = pure([value])
    target = boxed_type(target)
    if payload == target:
        return [result]
    result_types = make_result_types([target])
    with bind([result], result_types=result_types) as extracted:
        [value] = extracted.arguments
        if isinstance(
            target, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
        ) and not isinstance(value.type, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)):
            value = tvm_ffi_d.upcast(target, value)
        else:
            value = tvm_ffi_d.downcast(target, value)
        tvm_ffi_d.MonadYieldOp(pure([value], result_types=result_types))
    return [*extracted.outputs]


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if _is_numeric_scalar(lhs.type) and _is_numeric_scalar(rhs.type):
        return binary(operator.eq, lhs, rhs)
    else:
        return tvm_ffi_d.eq(box(lhs), box(rhs))


def ffi_type(name: str) -> ir.Type:
    """Parse a TVM FFI type description, e.g. ``int`` or ``monad.result<!tvm_ffi.int>``."""
    return ir.Type.parse(f"!tvm_ffi.{name}")


def is_binary_operation(operation: Any) -> bool:
    """Whether a callable has a native binary scalar implementation."""
    return operation in _BINARY_ARITHMETIC


def make_result_types(payload_types: Iterable[ir.Type]) -> list[ir.Type]:
    """Box scalar payload types and wrap them, using None for an empty sequence."""
    return [tvm_ffi_d.ResultType.get(boxed_type(type_)) for type_ in payload_types] or [
        tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
    ]


def native(value: ir.Value) -> ir.Value:
    return (
        tvm_ffi_d.get(value)
        if isinstance(
            value.type, (tvm_ffi_d.BoolType, tvm_ffi_d.IntType, tvm_ffi_d.FloatType)
        )
        else value
    )


def native_constant(value: bool | float) -> ir.Value:
    if isinstance(value, bool):
        type_ = ir.IntegerType.get_signless(1)
    elif isinstance(value, int):
        type_ = ir.IntegerType.get_signless(64)
        assert -(1 << 63) <= value < (1 << 63), (
            f"integer does not fit the i64 runtime ABI: {value}"
        )
    else:
        type_ = ir.F64Type.get()
    return arith.constant(type_, value)


def native_type(type_: ir.Type) -> ir.Type:
    if isinstance(type_, tvm_ffi_d.BoolType):
        return ir.IntegerType.get_signless(1)
    elif isinstance(type_, tvm_ffi_d.IntType):
        return ir.IntegerType.get_signless(64)
    elif isinstance(type_, tvm_ffi_d.FloatType):
        return ir.F64Type.get()
    else:
        return type_


def pure(
    values: Sequence[ir.Value], *, result_types: Sequence[ir.Type] | None = None
) -> list[ir.Value]:
    """Wrap borrowed payloads in successful Results with independent references."""
    assert all(is_value_in_scope(value) for value in values), (
        "pure payload is outside its defining block"
    )
    if result_types is None:
        result_types = make_result_types(value.type for value in values)
    assert result_types and all(
        isinstance(type_, tvm_ffi_d.ResultType) for type_ in result_types
    )
    assert len(result_types) == (len(values) or 1), (
        "payload count must match Result count"
    )
    if not values:
        assert isinstance(result_types[0].payload_type, tvm_ffi_d.NoneType)
        values = [constant(None)]
    payloads: list[ir.Value] = []
    for value, result in zip(values, result_types, strict=True):
        target = result.payload_type
        value = box(value)
        assert value.type == target or isinstance(
            target, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
        ), "payload type must match Result type"
        value = tvm_ffi_d.upcast(target, value)
        payloads.append(value)
    operation = tvm_ffi_d.MonadPureOp(result_types, payloads)

    return [*operation.outputs]


def tuple_type(types: Sequence[ir.Type]) -> tvm_ffi_d.TupleType:
    return tvm_ffi_d.TupleType.get([boxed_type(type_) for type_ in types])


def value_type(value: Any) -> ir.Type:
    if isinstance(value, torch.Tensor):
        return tvm_ffi_d.TensorType.get()
    elif isinstance(value, (bool, torch.SymBool)):
        return tvm_ffi_d.BoolType.get()
    elif isinstance(value, (int, torch.SymInt)):
        return tvm_ffi_d.IntType.get()
    elif isinstance(value, (float, torch.SymFloat)):
        return tvm_ffi_d.FloatType.get()
    elif isinstance(value, str):
        return tvm_ffi_d.StrType.get()
    elif isinstance(value, torch.device):
        return tvm_ffi_d.DeviceType.get()
    elif isinstance(value, torch.dtype):
        return tvm_ffi_d.DTypeType.get()
    elif isinstance(value, list):
        return array_type([value_type(item) for item in value])
    elif isinstance(value, tuple):
        return tuple_type([value_type(item) for item in value])
    elif isinstance(value, dict):
        return tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get())
    elif value is None:
        return tvm_ffi_d.NoneType.get()
    else:
        raise NotImplementedError(f"unsupported FX value type: {type(value)}")
