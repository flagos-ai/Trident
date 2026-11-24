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
from trident.core.dialects.tvm_ffi import arith as tvm_ffi_arith
from trident.core.dialects.tvm_ffi import abi as tvm_ffi_abi
from trident.core.dialects.tvm_ffi import api as tvm_ffi_api

_NUMERIC_SCALAR_TYPES = (
    tvm_ffi_abi.BoolType,
    tvm_ffi_abi.FloatType,
    tvm_ffi_abi.IntType,
)


# Result inference and numeric promotion belong to the TVM FFI dialect.
BINARY_ARITHMETIC: dict[
    Callable[..., Any], Callable[[ir.Value, ir.Value], ir.Value]
] = {
    max: tvm_ffi_arith.max,
    min: tvm_ffi_arith.min,
    operator.add: tvm_ffi_arith.add,
    operator.and_: tvm_ffi_arith.and_,
    operator.eq: tvm_ffi_arith.eq,
    operator.floordiv: tvm_ffi_arith.floordiv,
    operator.ge: tvm_ffi_arith.ge,
    operator.gt: tvm_ffi_arith.gt,
    operator.le: tvm_ffi_arith.le,
    operator.lshift: tvm_ffi_arith.lshift,
    operator.lt: tvm_ffi_arith.lt,
    operator.mod: tvm_ffi_arith.mod,
    operator.mul: tvm_ffi_arith.mul,
    operator.ne: tvm_ffi_arith.ne,
    operator.or_: tvm_ffi_arith.or_,
    operator.rshift: tvm_ffi_arith.rshift,
    operator.sub: tvm_ffi_arith.sub,
    operator.truediv: tvm_ffi_arith.truediv,
    operator.xor: tvm_ffi_arith.xor,
}


def constant(value: Any) -> ir.Value:
    if isinstance(value, bool):
        return tvm_ffi_abi.constant_bool(value)
    elif isinstance(value, int):
        assert -(1 << 63) <= value < (1 << 63), (
            f"integer does not fit the i64 runtime ABI: {value}"
        )
        return tvm_ffi_abi.constant_int(value)
    elif isinstance(value, float):
        return tvm_ffi_abi.constant_float(value)
    elif value is None:
        return tvm_ffi_abi.constant_none()
    elif isinstance(value, str):
        return tvm_ffi_api.constant_str(value)
    elif isinstance(value, torch.device):
        return tvm_ffi_abi.constant_device(f"{value}")
    elif isinstance(value, torch.dtype):
        dtype = runtime.convert(value)
        i64 = ir.IntegerType.get_signless(64)
        return tvm_ffi_abi.constant_dtype(
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
        return tvm_ffi_api.tensor_literal(elements)
    else:
        raise NotImplementedError(f"unsupported FX constant: {value!r}")


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if isinstance(lhs.type, _NUMERIC_SCALAR_TYPES) and isinstance(
        rhs.type, _NUMERIC_SCALAR_TYPES
    ):
        return tvm_ffi_arith.eq(lhs, rhs)
    else:
        return tvm_ffi_abi.to(tvm_ffi_api.eq(lhs, rhs))


def native(value: ir.Value) -> ir.Value:
    """Extract a temporary native scalar for a dialect interface."""
    assert isinstance(value.type, _NUMERIC_SCALAR_TYPES), (
        "scalar must use a TVM FFI type"
    )
    return tvm_ffi_abi.get(value)
