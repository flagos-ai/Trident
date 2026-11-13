# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Final

import torch._guards
from torch._guards import GuardSource

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


def parse_guards(guards: Iterable[torch._guards.Guard]) -> tuple[GuardCode, ...]:
    """Filter, parse, deduplicate and order one candidate's guard recipes."""
    # Captured globals and defaults are resolved by export, not dispatcher inputs.
    unique: dict[Hashable, GuardCode] = {}
    for guard in guards:
        if guard.source not in _CAPTURE_GUARD_SOURCES:
            parsed = Guard.parse(guard)
            assert parsed is not None, (
                "unsupported Dynamo guard: "
                f"{guard.create_fn_name()} {guard.name!r} {guard.code_list!r}"
            )
            for code in parsed.codes:
                unique[code.key] = code
    # Stable sorting preserves capture order within each priority.
    return tuple(sorted(unique.values(), key=lambda code: code.priority))


__all__ = [
    "parse_guards",
]
