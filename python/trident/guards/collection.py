# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import Final

import torch._guards
from torch._guards import GuardSource

from trident.core import ir
from trident.core.dialects import scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.input import InputTableBuilder
from trident.ir import bind, native, native_constant, pure
from trident.ownership import RETAIN_ATTR

from .codes import GuardCode
from .kinds import Guard

_CAPTURE_GUARD_SOURCES: Final[frozenset[GuardSource]] = frozenset(
    {
        GuardSource.CONSTANT,
        GuardSource.GLOBAL,
        GuardSource.GLOBAL_FSDP_MODULE,
        GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
        GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    }
)


class Guards:
    """Build a short-circuiting IR chain from a Torch ``GuardsSet``."""

    def __init__(self, guards: Iterable[torch._guards.Guard]) -> None:
        self.guards: Final[list[torch._guards.Guard]] = [*guards]

    def _build_checks(
        self,
        codes: Iterator[GuardCode],
        builder: InputTableBuilder,
        arguments: Sequence[ir.Value],
        context: ir.Context,
    ) -> list[ir.Value]:
        type_ = tvm_ffi_d.ResultType.get(tvm_ffi_d.BoolType.get())
        code = next(codes, None)
        if code is None:
            return pure([native_constant(True)], result_types=[type_])
        with bind(
            code.build_result(builder.build(arguments), context),
            result_types=[type_],
        ) as operation:
            [predicate] = operation.arguments
            branch = scf.IfOp(native(predicate), [type_], has_else=True)
            with ir.InsertionPoint(branch.then_block):
                yielded = self._build_checks(codes, builder, arguments, context)
                terminator = scf.YieldOp(yielded)
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            with ir.InsertionPoint(branch.else_block):
                yielded = pure([native_constant(False)], result_types=[type_])
                terminator = scf.YieldOp(yielded)
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            terminator = tvm_ffi_d.MonadYieldOp(branch.results)
            terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
        return [*operation.outputs]

    def build(
        self,
        builder: InputTableBuilder,
        arguments: Sequence[ir.Value],
        context: ir.Context,
    ) -> list[ir.Value]:
        # Captured globals and defaults are resolved by export, not wrapper inputs.
        unique: dict[Hashable, GuardCode] = {}
        for guard in self.guards:
            if guard.source not in _CAPTURE_GUARD_SOURCES:
                parsed = Guard.parse(guard)
                assert parsed is not None, (
                    "unsupported Dynamo guard: "
                    f"{guard.create_fn_name()} {guard.name!r} {guard.code_list!r}"
                )
                for code in parsed.codes:
                    unique[code.key] = code
        ordered: list[GuardCode] = sorted(
            unique.values(),
            key=lambda code: (
                # Lower priorities run first; depth orders checks within each
                # priority, with nested structure checks running outside-in.
                code.priority,
                code.depth,
            ),
        )

        return self._build_checks(iter(ordered), builder, arguments, context)
