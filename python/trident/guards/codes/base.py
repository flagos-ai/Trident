# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from abc import abstractmethod
from collections.abc import Hashable
from typing import Final

from typing_extensions import Self

from trident.core import ir
from trident.ir import pure

from ..scope import Scope


class GuardCode:
    """Read-only check whose result may be shared within one dispatch."""

    # Metadata and constant checks precede general expression checks.
    priority: int = 1

    def __init__(self, text: str, source: ast.Subscript | None) -> None:
        self.text: Final[str] = text
        self.source: Final[ast.Subscript | None] = source

    @abstractmethod
    def build(
        self,
        scope: Scope,
        context: ir.Context,
    ) -> ir.Value:
        raise NotImplementedError(
            f"guard code {type(self).__name__} cannot be lowered: {self.text!r}"
        )

    def build_result(self, scope: Scope, context: ir.Context) -> list[ir.Value]:
        return [pure(self.build(scope, context))]

    @property
    def key(self) -> Hashable:
        return (
            type(self),
            None if self.source is None else ast.dump(self.source),
            self.text,
        )

    @classmethod
    @abstractmethod
    def parse(
        cls,
        text: str,
        source: ast.Subscript | None,
    ) -> Self | None: ...
