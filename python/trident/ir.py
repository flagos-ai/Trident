# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Shared IR builders for graph import, input handling and Dynamo guards.

Frontend values use TVM FFI types. Native scalars are temporary operands of
local arithmetic or native dialect interfaces, never stored frontend values.
Checked calls validate lookup status, function handles and call status.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from functools import partial
from types import TracebackType
from typing import Any, Self, TypeVar

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import arith, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d


def _is_numeric_scalar(type_: ir.Type) -> bool:
    return isinstance(
        type_, (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
    )


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


def array_type(types: Sequence[ir.Type]) -> tvm_ffi_d.ArrayType:
    elements = [*types]
    element = (
        elements[0]
        if elements and all(type_ == elements[0] for type_ in elements)
        else tvm_ffi_d.AnyType.get()
    )
    return tvm_ffi_d.ArrayType.get(element)


def binary(operation: Any, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    assert all(_is_numeric_scalar(value.type) for value in (lhs, rhs)), (
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
        if not keep_bool and _is_signless_integer(value.type, 1):
            value = arith.extui(ir.IntegerType.get_signless(64), value)
        if use_float and not isinstance(value.type, ir.F64Type):
            value = arith.sitofp(ir.F64Type.get(), value)
        operands.append(value)
    return tvm_ffi_d.to(builder(*operands))


class Scope(ABC):
    """Construct one block with an explicit, operation-specific terminator."""

    def __init__(self, block: ir.Block) -> None:
        self.block = block
        self._insertion_point = ir.InsertionPoint(block)

    def __enter__(self) -> Self:
        self._insertion_point.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._insertion_point.__exit__(exc_type, exc_value, traceback)

    @property
    @abstractmethod
    def results(self) -> list[ir.Value]:
        """Complete operation results forwarded to the enclosing scope."""

    @abstractmethod
    def yield_(self, values: Sequence[ir.Value]) -> None:
        """Terminate only this block; operation results belong to its owner."""


class BindScope(Scope):
    """Expose borrowed payloads in one monadic success region."""

    def __init__(
        self, result_types: Sequence[ir.Type], inputs: Sequence[ir.Value]
    ) -> None:
        assert all(isinstance(type_, tvm_ffi_d.ResultType) for type_ in result_types)
        assert all(isinstance(value.type, tvm_ffi_d.ResultType) for value in inputs), (
            "bind inputs must be Results"
        )
        self.operation = tvm_ffi_d.MonadBindOp(result_types, inputs)
        super().__init__(self.operation.add_entry_block())

    @property
    def arguments(self) -> Sequence[ir.Value]:
        return self.operation.arguments

    @property
    def results(self) -> list[ir.Value]:
        return [*self.operation.outputs]

    def yield_(self, values: Sequence[ir.Value]) -> None:
        with ir.InsertionPoint(self.block):
            tvm_ffi_d.monad_yield(values)


_ScopeT = TypeVar("_ScopeT", bound=Scope)


class ScopeStack(ExitStack):
    """Enter local scopes and explicitly terminate them from innermost outward."""

    def __init__(self) -> None:
        super().__init__()
        self._scopes: list[Scope] = []

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self._scopes.clear()

    def bind(
        self, result_types: Sequence[ir.Type], inputs: Sequence[ir.Value]
    ) -> Sequence[ir.Value]:
        return self.push(BindScope(result_types, inputs)).arguments

    def push(self, scope: _ScopeT) -> _ScopeT:
        self.enter_context(scope)
        self._scopes.append(scope)
        return scope

    def yield_(self, values: Sequence[ir.Value]) -> list[ir.Value]:
        """Drain this stack, forwarding complete results from each scope."""
        results = [*values]
        while self._scopes:
            scope = self._scopes.pop()
            scope.yield_(results)
            results = scope.results
        self.close()
        return results


def checked_call(
    name: str, arguments: Sequence[ir.Value], result: ir.Type
) -> list[ir.Value]:
    """Return an asserting invocation wrapped as a successful Result."""
    handle = tvm_ffi_d.FunctionGetGlobalOp(name).result
    call = tvm_ffi_d.FunctionCallOp(result, handle, arguments)
    return pure([call.result])


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


def constant_type(value: Any) -> ir.Type:
    """TVM FFI type produced by a leaf constant."""
    return value_type(value)


def container_create(values: Sequence[ir.Value], type_: ir.Type) -> list[ir.Value]:
    elements = [*values]
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
        value = tvm_ffi_d.tuple_construct(type_, elements)
    else:
        value = tvm_ffi_d.array_create(type_, elements)
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
        value = tvm_ffi_d.array_get(
            container, native(constant(index) if isinstance(index, int) else index)
        )
    else:
        return checked_call(
            "ffi.ArrayGetItem",
            [container, constant(index) if isinstance(index, int) else index],
            target,
        )
    widening = isinstance(
        target, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
    ) and not isinstance(payload, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType))
    if payload != target and not widening:
        value = tvm_ffi_d.downcast(target, value)
    return pure([value], result_types=make_result_types([target]))


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if _is_numeric_scalar(lhs.type) and _is_numeric_scalar(rhs.type):
        return binary(operator.eq, lhs, rhs)
    else:
        return tvm_ffi_d.to(tvm_ffi_d.eq(lhs, rhs))


def ffi_type(name: str) -> ir.Type:
    """Parse a TVM FFI type description, e.g. ``int`` or ``monad.result<!tvm_ffi.int>``."""
    return ir.Type.parse(f"!tvm_ffi.{name}")


def is_binary_operation(operation: Any) -> bool:
    """Whether a callable has a native binary scalar implementation."""
    return operation in _BINARY_ARITHMETIC


def logical_not(value: ir.Value) -> ir.Value:
    """Negate a numeric truth value, retaining a semantic Bool result."""
    return equal(value, constant(0))


def make_result_types(payload_types: Iterable[ir.Type]) -> list[ir.Type]:
    """Wrap TVM FFI payload types, using None for an empty sequence."""
    return [tvm_ffi_d.ResultType.get(type_) for type_ in payload_types] or [
        tvm_ffi_d.ResultType.get(tvm_ffi_d.NoneType.get())
    ]


def native(value: ir.Value) -> ir.Value:
    """Extract a temporary native scalar for arithmetic or a dialect interface."""
    assert _is_numeric_scalar(value.type), "scalar must use a TVM FFI type"
    return tvm_ffi_d.get(value)


def negate(value: ir.Value) -> ir.Value:
    if isinstance(value.type, tvm_ffi_d.FloatType):
        return tvm_ffi_d.to(arith.negf(native(value)))
    return binary(operator.sub, constant(0), value)


def pure(
    values: Sequence[ir.Value], *, result_types: Sequence[ir.Type] | None = None
) -> list[ir.Value]:
    """Wrap borrowed payloads in successful Results with independent references."""
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
        assert value.type == target or isinstance(
            target, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
        ), "payload type must match Result type"
        value = tvm_ffi_d.upcast(target, value)
        payloads.append(value)
    operation = tvm_ffi_d.MonadPureOp(result_types, payloads)

    return [*operation.outputs]


def select(condition: ir.Value, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    """Select between equally typed semantic values without exposing native results."""
    assert isinstance(condition.type, tvm_ffi_d.BoolType)
    assert lhs.type == rhs.type, "select alternatives must have the same TVM FFI type"
    selection = scf.IfOp(native(condition), [lhs.type], has_else=True)
    with ir.InsertionPoint(selection.then_block):
        scf.yield_([lhs])
    with ir.InsertionPoint(selection.else_block):
        scf.yield_([rhs])
    [result] = selection.results
    return result


def tensor_device_fields(tensor: ir.Value) -> list[ir.Value]:
    """Read the DLPack device type and index as semantic integer values."""
    return [
        tvm_ffi_d.to(arith.extsi(ir.IntegerType.get_signless(64), field))
        for field in tvm_ffi_d.tensor_device(tensor)
    ]


def tensor_dim(tensor: ir.Value) -> ir.Value:
    return tvm_ffi_d.to(tvm_ffi_d.tensor_dim(tensor))


def tensor_dtype_fields(tensor: ir.Value) -> list[ir.Value]:
    """Read unsigned DLPack dtype fields as semantic integer values."""
    return [
        tvm_ffi_d.to(arith.extui(ir.IntegerType.get_signless(64), field))
        for field in tvm_ffi_d.tensor_dtype(tensor)
    ]


def tensor_size(tensor: ir.Value, index: ir.Value) -> ir.Value:
    return tvm_ffi_d.to(tvm_ffi_d.tensor_size(tensor, native(index)))


def tensor_storage_offset(tensor: ir.Value) -> ir.Value:
    return tvm_ffi_d.to(tvm_ffi_d.tensor_storage_offset(tensor))


def tensor_stride(tensor: ir.Value, index: ir.Value) -> ir.Value:
    return tvm_ffi_d.to(tvm_ffi_d.tensor_stride(tensor, native(index)))


def tuple_type(types: Sequence[ir.Type]) -> tvm_ffi_d.TupleType:
    return tvm_ffi_d.TupleType.get(types)


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
