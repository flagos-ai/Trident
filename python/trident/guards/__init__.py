# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Final

import torch._guards
from torch._guards import GuardSource

from .codes import GuardCode, parse_code
from .source import parse_source

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

_IGNORED_GUARDS: Final[frozenset[str]] = frozenset(
    {
        "AUTOGRAD_SAVED_TENSORS_HOOKS",
        "DEFAULT_DEVICE",
        "DETERMINISTIC_ALGORITHMS",
        "GLOBAL_STATE",
        "GRAD_MODE",
        "TORCH_FUNCTION_STATE",
    }
)


def parse_guards(guards: Iterable[torch._guards.Guard]) -> tuple[GuardCode, ...]:
    """Filter, parse, deduplicate and order one candidate's guard recipes."""
    # Captured globals and defaults are resolved by export, not dispatcher inputs.
    unique: dict[Hashable, GuardCode] = {}
    for guard in guards:
        if (
            guard.source in _CAPTURE_GUARD_SOURCES
            or guard.create_fn_name() in _IGNORED_GUARDS
        ):
            continue
        # Dynamo emits an empty shape environment for graphs without shape checks.
        if guard.create_fn_name() == "SHAPE_ENV" and not guard.code_list:
            continue
        diagnostic = (
            "unsupported Dynamo guard: "
            f"{guard.create_fn_name()} {guard.name!r} {guard.code_list!r}"
        )
        assert guard.code_list, diagnostic
        source = parse_source(guard.name)
        for text in guard.code_list:
            code = parse_code(text, source)
            assert code is not None, f"{diagnostic}; unsupported code: {text!r}"
            unique[code.key] = code
    # Stable sorting preserves capture order within each priority.
    return tuple(sorted(unique.values(), key=lambda code: code.priority))


__all__ = [
    "parse_guards",
]
