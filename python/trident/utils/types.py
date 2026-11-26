# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Type inference for Python values and semantic containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from trident.core import ir
from trident.core.dialects import tvm_ffi


def array_type(types: Sequence[ir.Type]) -> tvm_ffi.api.ArrayType:
    elements = set(types)
    if len(elements) == 1:
        (element,) = elements
    else:
        element = tvm_ffi.api.AnyType.get()
    return tvm_ffi.api.ArrayType.get(element)


def value_type(value: Any) -> ir.Type:
    if isinstance(value, torch.Tensor):
        return tvm_ffi.api.TensorType.get()
    elif isinstance(value, (bool, torch.SymBool)):
        return tvm_ffi.abi.BoolType.get()
    elif isinstance(value, (int, torch.SymInt)):
        return tvm_ffi.abi.IntType.get()
    elif isinstance(value, (float, torch.SymFloat)):
        return tvm_ffi.abi.FloatType.get()
    elif isinstance(value, str):
        return tvm_ffi.api.StrType.get()
    elif isinstance(value, torch.device):
        return tvm_ffi.abi.DeviceType.get()
    elif isinstance(value, torch.dtype):
        return tvm_ffi.abi.DTypeType.get()
    elif isinstance(value, list):
        return array_type([value_type(item) for item in value])
    elif isinstance(value, tuple):
        return tvm_ffi.api.TupleType.get([value_type(item) for item in value])
    elif isinstance(value, dict):
        return tvm_ffi.api.ArrayType.get(tvm_ffi.api.AnyType.get())
    elif value is None:
        return tvm_ffi.abi.NoneType.get()
    else:
        raise NotImplementedError(f"unsupported FX value type: {type(value)}")
