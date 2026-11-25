# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Semantic constants and boxed scalar arithmetic builders."""

from __future__ import annotations

from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import tvm_ffi

_NUMERIC_SCALAR_TYPES = (
    tvm_ffi.abi.BoolType,
    tvm_ffi.abi.FloatType,
    tvm_ffi.abi.IntType,
)


def arithmetic(name: str, *operands: ir.Value) -> ir.Value:
    """Build scalar arithmetic with dialect-owned inference and promotion."""
    operation = ir.Operation.create(
        f"tvm_ffi_arith.{name}", operands=operands, infer_type=True
    )
    [result] = operation.results
    return result


def constant(value: Any) -> ir.Value:
    if isinstance(value, bool):
        return tvm_ffi.abi.constant_bool(value)
    elif isinstance(value, int):
        assert -(1 << 63) <= value < (1 << 63), (
            f"integer does not fit the i64 runtime ABI: {value}"
        )
        return tvm_ffi.abi.constant_int(value)
    elif isinstance(value, float):
        return tvm_ffi.abi.constant_float(value)
    elif value is None:
        return tvm_ffi.abi.constant_none()
    elif isinstance(value, str):
        return tvm_ffi.api.constant_str(value)
    elif isinstance(value, torch.device):
        return tvm_ffi.abi.constant_device(f"{value}")
    elif isinstance(value, torch.dtype):
        dtype = runtime.convert(value)
        i64 = ir.IntegerType.get_signless(64)
        return tvm_ffi.abi.constant_dtype(
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
        return tvm_ffi.api.tensor_literal(elements)
    else:
        raise NotImplementedError(f"unsupported FX constant: {value!r}")


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if isinstance(lhs.type, _NUMERIC_SCALAR_TYPES) and isinstance(
        rhs.type, _NUMERIC_SCALAR_TYPES
    ):
        return tvm_ffi.arith.eq(lhs, rhs)
    else:
        return tvm_ffi.abi.to(tvm_ffi.api.eq(lhs, rhs))


def native(value: ir.Value) -> ir.Value:
    """Extract a temporary native scalar for a dialect interface."""
    assert isinstance(value.type, _NUMERIC_SCALAR_TYPES), (
        "scalar must use a TVM FFI type"
    )
    return tvm_ffi.abi.get(value)
