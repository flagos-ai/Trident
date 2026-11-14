# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Semantic constants and boxed scalar arithmetic builders."""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import arith
from trident.core.dialects import tvm_ffi as tvm_ffi_d

_NUMERIC_SCALAR_TYPES = (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)


def _is_signless_integer(type_: ir.Type, width: int) -> bool:
    return (
        isinstance(type_, ir.IntegerType) and type_.is_signless and type_.width == width
    )


# Entries contain integer and floating-point implementations, in that order.
_BINARY_ARITHMETIC: dict[
    Callable[..., Any],
    tuple[
        Callable[[ir.Value, ir.Value], ir.Value] | None,
        Callable[[ir.Value, ir.Value], ir.Value] | None,
    ],
] = {
    max: (arith.maxsi, arith.maximumf),
    min: (arith.minsi, arith.minimumf),
    operator.add: (arith.addi, arith.addf),
    operator.and_: (arith.andi, None),
    operator.eq: (
        partial(arith.cmpi, arith.CmpIPredicate.eq),
        partial(arith.cmpf, arith.CmpFPredicate.OEQ),
    ),
    operator.floordiv: (arith.floordivsi, None),
    operator.ge: (
        partial(arith.cmpi, arith.CmpIPredicate.sge),
        partial(arith.cmpf, arith.CmpFPredicate.OGE),
    ),
    operator.gt: (
        partial(arith.cmpi, arith.CmpIPredicate.sgt),
        partial(arith.cmpf, arith.CmpFPredicate.OGT),
    ),
    operator.le: (
        partial(arith.cmpi, arith.CmpIPredicate.sle),
        partial(arith.cmpf, arith.CmpFPredicate.OLE),
    ),
    operator.lshift: (arith.shli, None),
    operator.lt: (
        partial(arith.cmpi, arith.CmpIPredicate.slt),
        partial(arith.cmpf, arith.CmpFPredicate.OLT),
    ),
    operator.mod: (
        lambda lhs, rhs: arith.subi(lhs, arith.muli(arith.floordivsi(lhs, rhs), rhs)),
        None,
    ),
    operator.mul: (arith.muli, arith.mulf),
    operator.ne: (
        partial(arith.cmpi, arith.CmpIPredicate.ne),
        partial(arith.cmpf, arith.CmpFPredicate.UNE),
    ),
    operator.or_: (arith.ori, None),
    operator.rshift: (arith.shrsi, None),
    operator.sub: (arith.subi, arith.subf),
    operator.truediv: (None, arith.divf),
    operator.xor: (arith.xori, None),
}


def binary(operation: Any, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    assert all(isinstance(value.type, _NUMERIC_SCALAR_TYPES) for value in (lhs, rhs)), (
        "scalar operands must use TVM FFI types"
    )
    values = (native(lhs), native(rhs))
    integer, floating = _BINARY_ARITHMETIC[operation]
    use_float = operation is operator.truediv or any(
        isinstance(value.type, ir.F64Type) for value in values
    )
    builder = floating if use_float else integer
    if builder is None:
        raise NotImplementedError(f"unsupported scalar operation: {operation}")
    keep_bool = operation in (operator.and_, operator.or_, operator.xor) and all(
        _is_signless_integer(value.type, 1) for value in values
    )
    operands: list[ir.Value] = []
    for value in values:
        if use_float and not isinstance(value.type, ir.F64Type):
            value = (
                arith.uitofp(ir.F64Type.get(), value)
                if _is_signless_integer(value.type, 1)
                else arith.sitofp(ir.F64Type.get(), value)
            )
        elif not keep_bool and _is_signless_integer(value.type, 1):
            value = arith.extui(ir.IntegerType.get_signless(64), value)
        operands.append(value)
    return tvm_ffi_d.to(builder(*operands))


BINARY_ARITHMETIC: dict[
    Callable[..., Any], Callable[[ir.Value, ir.Value], ir.Value]
] = {operation: partial(binary, operation) for operation in _BINARY_ARITHMETIC}


def constant(value: Any) -> ir.Value:
    if isinstance(value, bool):
        return tvm_ffi_d.constant_bool(value)
    elif isinstance(value, int):
        assert -(1 << 63) <= value < (1 << 63), (
            f"integer does not fit the i64 runtime ABI: {value}"
        )
        return tvm_ffi_d.constant_int(value)
    elif isinstance(value, float):
        return tvm_ffi_d.constant_float(value)
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
        return tvm_ffi_d.tensor_literal(elements)
    else:
        raise NotImplementedError(f"unsupported FX constant: {value!r}")


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if isinstance(lhs.type, _NUMERIC_SCALAR_TYPES) and isinstance(
        rhs.type, _NUMERIC_SCALAR_TYPES
    ):
        return binary(operator.eq, lhs, rhs)
    else:
        return tvm_ffi_d.to(tvm_ffi_d.eq(lhs, rhs))


def logical_not(value: ir.Value) -> ir.Value:
    """Compare against zero, preserving NaN truthiness and signed-zero behavior."""
    return equal(value, constant(0))


def native(value: ir.Value) -> ir.Value:
    """Extract a temporary native scalar for arithmetic or a dialect interface."""
    assert isinstance(value.type, _NUMERIC_SCALAR_TYPES), (
        "scalar must use a TVM FFI type"
    )
    return tvm_ffi_d.get(value)


def negate(value: ir.Value) -> ir.Value:
    return (
        tvm_ffi_d.to(arith.negf(native(value)))
        if isinstance(value.type, tvm_ffi_d.FloatType)
        else binary(operator.sub, constant(0), value)
    )
