# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Shared direct reads from semantic containers."""

from __future__ import annotations

from trident.core import ir
from trident.core.dialects.tvm_ffi import api as tvm_ffi_api

from .values import constant, native


def container_read(
    container: ir.Value, index: int | ir.Value, target: ir.Type
) -> ir.Value:
    """Read a semantic container directly, assuming a valid container and index."""
    type_ = container.type
    assert isinstance(type_, (tvm_ffi_api.ArrayType, tvm_ffi_api.TupleType))
    if isinstance(type_, tvm_ffi_api.TupleType):
        assert isinstance(index, int)
        assert -len(type_.types) <= index < len(type_.types), "tuple index out of range"
        payload = type_.types[index]
        value = tvm_ffi_api.TupleUnconstructOp(type_.types, container).elements[index]
    else:
        payload = type_.element_type
        value = tvm_ffi_api.array_get(
            container, native(constant(index) if isinstance(index, int) else index)
        )
    widening = isinstance(
        target, (tvm_ffi_api.AnyType, tvm_ffi_api.UnionType)
    ) and not isinstance(payload, (tvm_ffi_api.AnyType, tvm_ffi_api.UnionType))
    if payload != target and not widening:
        value = tvm_ffi_api.downcast(target, value)
    return (
        tvm_ffi_api.upcast(target, value) if payload != target and widening else value
    )
