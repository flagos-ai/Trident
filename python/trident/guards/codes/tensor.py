# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Hashable
from functools import partial, reduce
from typing import Final, Self, override

import torch
import tvm_ffi

from trident.core import ir
from trident.input import InputTable
from trident.ir import (
    binary,
    constant,
    equal,
    tensor_device_fields,
    tensor_dim,
    tensor_dtype_fields,
)

from ..local import Local
from .base import GuardCode


class DynamoAttributeAbsentCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        attribute: str,
    ) -> None:
        super().__init__(text, source)
        self.attribute: Final[str] = attribute

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        # Private Dynamo attributes are not represented by the TVM FFI ABI.
        return constant(True)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*hasattr\(\s*{source.regex}\s*,\s*"
            rf"(?P<quote>['\"])(?P<attribute>_dynamo_[A-Za-z0-9_]*)"
            rf"(?P=quote)\s*\)\s*==\s*False\s*",
            text,
        )
        return (
            cls(text, source, match.group("attribute")) if match is not None else None
        )


class RequiresGradCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expected: bool,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[bool] = expected

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        # The tensor ABI does not expose autograd state.
        return constant(True)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*{source.regex}\s*\.requires_grad\s*==\s*"
            rf"(?P<value>True|False)\s*",
            text,
        )
        if match is None:
            return None
        else:
            return cls(text, source, match.group("value") == "True")


class TensorDTypeCode(GuardCode):
    requires_input = True

    def __init__(
        self,
        text: str,
        source: Local,
        expected: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[str] = expected
        self.dtype: Final[torch.dtype] = dtype

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        source = self.source
        assert source is not None, f"guard source is required: {self.text!r}"
        tensor = source.resolve(tree)
        assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
        metadata = tensor_dtype_fields(tensor)
        dtype = tvm_ffi.convert(self.dtype)
        return reduce(
            partial(binary, operator.and_),
            [
                equal(actual, constant(expected_value))
                for actual, expected_value in zip(
                    metadata, (dtype.type_code, dtype.bits, dtype.lanes), strict=True
                )
            ],
            constant(True),
        )

    @property
    def key(self) -> Hashable:
        return ("tensor-dtype", self.source, self.expected)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*str\(\s*{source.regex}\s*\.dtype\s*\)\s*==\s*"
            rf"(?P<quote>['\"])(?P<dtype>torch\.[A-Za-z_][A-Za-z0-9_]*)"
            rf"(?P=quote)\s*",
            text,
        )
        if match is None:
            return None
        expected = match.group("dtype")
        dtype = getattr(torch, expected.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            return None
        else:
            return cls(text, source, expected, dtype)


class TensorDeviceCode(GuardCode):
    requires_input = True

    def __init__(
        self,
        text: str,
        source: Local,
        expected: str,
        device: torch.device,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[str] = expected
        self.device: Final[torch.device] = device

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        source = self.source
        assert source is not None, f"guard source is required: {self.text!r}"
        tensor = source.resolve(tree)
        assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
        device_type, device_index = tensor_device_fields(tensor)
        return binary(
            operator.and_,
            equal(
                device_type,
                constant(tvm_ffi.device(self.device.type).dlpack_device_type()),
            ),
            equal(
                device_index,
                constant(0 if self.device.index is None else self.device.index),
            ),
        )

    @property
    def key(self) -> Hashable:
        return ("tensor-device", self.source, self.expected)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*str\(\s*{source.regex}\s*\.device\s*\)\s*==\s*"
            rf"(?P<quote>['\"])(?P<device>[^'\"]+)(?P=quote)\s*",
            text,
        )
        if match is None:
            return None
        expected = match.group("device")
        device = torch.device(expected)
        return cls(text, source, expected, device)


class TensorRankCode(GuardCode):
    requires_input = True

    def __init__(
        self,
        text: str,
        source: Local,
        expected: int,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[int] = expected

    @override
    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        source = self.source
        assert source is not None, f"guard source is required: {self.text!r}"
        tensor = source.resolve(tree)
        assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
        return equal(tensor_dim(tensor), constant(self.expected))

    @property
    def key(self) -> Hashable:
        return ("tensor-dim", self.source, self.expected)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*{source.regex}\s*\.ndimension\(\s*\)\s*==\s*"
            rf"(?P<rank>\d+)\s*",
            text,
        )
        if match is None:
            return None
        rank = ast.literal_eval(match.group("rank"))
        assert isinstance(rank, int) and not isinstance(rank, bool)
        return cls(text, source, rank)
