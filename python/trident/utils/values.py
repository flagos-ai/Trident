# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Semantic constants and boxed scalar arithmetic builders."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d

_NUMERIC_SCALAR_TYPES = (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)


# Result inference and numeric promotion belong to the TVM FFI dialect.
BINARY_ARITHMETIC: dict[
    Callable[..., Any], Callable[[ir.Value, ir.Value], ir.Value]
] = {
    max: tvm_ffi_d.arith_max,
    min: tvm_ffi_d.arith_min,
    operator.add: tvm_ffi_d.arith_add,
    operator.and_: tvm_ffi_d.arith_and,
    operator.eq: tvm_ffi_d.arith_eq,
    operator.floordiv: tvm_ffi_d.arith_floordiv,
    operator.ge: tvm_ffi_d.arith_ge,
    operator.gt: tvm_ffi_d.arith_gt,
    operator.le: tvm_ffi_d.arith_le,
    operator.lshift: tvm_ffi_d.arith_lshift,
    operator.lt: tvm_ffi_d.arith_lt,
    operator.mod: tvm_ffi_d.arith_mod,
    operator.mul: tvm_ffi_d.arith_mul,
    operator.ne: tvm_ffi_d.arith_ne,
    operator.or_: tvm_ffi_d.arith_or,
    operator.rshift: tvm_ffi_d.arith_rshift,
    operator.sub: tvm_ffi_d.arith_sub,
    operator.truediv: tvm_ffi_d.arith_truediv,
    operator.xor: tvm_ffi_d.arith_xor,
}


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
        return tvm_ffi_d.arith_eq(lhs, rhs)
    else:
        return tvm_ffi_d.to(tvm_ffi_d.eq(lhs, rhs))


def native(value: ir.Value) -> ir.Value:
    """Extract a temporary native scalar for a dialect interface."""
    assert isinstance(value.type, _NUMERIC_SCALAR_TYPES), (
        "scalar must use a TVM FFI type"
    )
    return tvm_ffi_d.get(value)
