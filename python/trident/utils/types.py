# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Type inference for Python values and semantic containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d


def array_type(types: Sequence[ir.Type]) -> tvm_ffi_d.ArrayType:
    elements = [*types]
    element = tvm_ffi_d.AnyType.get()
    if elements:
        first, *remaining = elements
        if all(type_ == first for type_ in remaining):
            element = first
    return tvm_ffi_d.ArrayType.get(element)


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
        return tvm_ffi_d.TupleType.get([value_type(item) for item in value])
    elif isinstance(value, dict):
        return tvm_ffi_d.ArrayType.get(tvm_ffi_d.AnyType.get())
    elif value is None:
        return tvm_ffi_d.NoneType.get()
    else:
        raise NotImplementedError(f"unsupported FX value type: {type(value)}")
