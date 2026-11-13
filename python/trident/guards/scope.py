# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import container_read, pure


class Scope:
    """A persistent tree containing only materialized SSA values."""

    def __init__(
        self,
        value: ir.Value,
        children: Mapping[int | str, Scope] | None = None,
    ) -> None:
        self.value: Final[ir.Value] = value
        self.children: Final[Mapping[int | str, Scope]] = dict(children or {})

    @property
    def type(self) -> ir.Type:
        return self.value.type

    def __getitem__(self, key: int | str) -> Scope:
        if (input_scope := self.get(key)) is not None:
            return input_scope
        else:
            raise KeyError(key)

    def get(self, key: int | str) -> Scope | None:
        """Return a materialized child, or None without generating IR."""
        return self.children.get(key)

    def child(self, key: int | str) -> tuple[Scope, Scope]:
        """Return the child and updated node, creating a missing child lazily."""
        if (input_scope := self.get(key)) is not None:
            return input_scope, self
        assert isinstance(key, int) and isinstance(
            self.value.type, (tvm_ffi_d.ArrayType, tvm_ffi_d.TupleType)
        )
        type_ = (
            self.value.type.types[key]
            if isinstance(self.value.type, tvm_ffi_d.TupleType)
            else self.value.type.element_type
        )
        value = container_read(self.value, key, type_)
        input_scope = Scope(value)
        scope = Scope(self.value, {**self.children, key: input_scope})
        return input_scope, scope

    def read(self, input_path: tuple[int | str, ...]) -> tuple[ir.Value, Scope]:
        """Read a path and reconnect materialized children to a persistent root."""
        if not input_path:
            return self.value, self
        key, *keys = input_path
        child, scope = self.child(key)
        value, updated_child = child.read(tuple(keys))
        if updated_child is not child:
            scope = Scope(scope.value, {**scope.children, key: updated_child})
        return value, scope

    def extract(self, input_path: tuple[int | str, ...]) -> tuple[ir.Value, Scope]:
        """Wrap a locally reusable payload in a fresh Result and return its root."""
        value, scope = self.read(input_path)
        return pure(value), scope
