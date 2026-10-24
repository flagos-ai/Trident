# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from typing import ClassVar, Final, Self

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.input import InputTable
from trident.ir import BindScope, pure

from ..local import Local


class GuardCode:
    """Parsed representation of one expression from a Guard code list."""

    # Lower priorities are built first.  Priority zero is reserved for
    # structure checks, whose source depth is used as a secondary key.
    priority: ClassVar[int] = 1
    requires_input: ClassVar[bool] = False

    def __init__(self, text: str, source: Local | None) -> None:
        self.text: Final[str] = text
        self.source: Final[Local | None] = source

    @abstractmethod
    def build(
        self,
        tree: InputTable,
        context: ir.Context,
    ) -> ir.Value:
        raise NotImplementedError(
            f"guard code {type(self).__name__} cannot be lowered: {self.text!r}"
        )

    def build_result(self, tree: InputTable, context: ir.Context) -> list[ir.Value]:
        types = [tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())]
        if self.source is None or not self.requires_input:
            return pure([self.build(tree, context)], result_types=types)
        with BindScope(types, tree.extract(self.source.subscripts)) as scope:
            payloads = scope.arguments
            [value] = payloads
            scoped = tree.with_value(self.source.subscripts, value)
            scope.yield_(pure([self.build(scoped, context)], result_types=types))
            bound_results = scope.results

        return [*bound_results]

    @property
    def depth(self) -> int:
        return 0

    @property
    def key(self) -> Hashable:
        return (type(self), self.source, self.text)

    @classmethod
    @abstractmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None: ...
