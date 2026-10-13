# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Deterministic inputs and eager-reference checks for individual operators."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import torch
import trident


def make_inputs(kind: str, device: torch.device, dtype: torch.dtype, seed: int) -> tuple:
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    def indices(values: list) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.int64)

    match kind:
        case "unary":
            return (rand(3, 4),)
        case "positive":
            return (rand(3, 4).abs() + 0.5,)
        case "binary":
            return rand(3, 4), rand(3, 4)
        case "broadcast":
            return rand(3, 4), rand(4)
        case "comparison":
            x, y = rand(3, 4), rand(4)
            # Include equal, less-than and greater-than values in every case.
            x[0] = y
            x[1] = y - 1
            x[2] = y + 1
            return x, y
        case "divisor":
            return rand(3, 4), rand(4).abs() + 0.5
        case "boolean":
            return rand(3, 4) > 0, rand(3, 4) > 0
        case "boolean_unary":
            x = rand(3, 4) > 0
            x[0] = False
            x[1] = True
            return (x,)
        case "where":
            return rand(3, 4) > 0, rand(3, 4), rand(4)
        case "noncontiguous":
            return (rand(4, 3).transpose(0, 1).clone(memory_format=torch.preserve_format),)
        case "singleton":
            return (rand(1, 3, 1, 4),)
        case "expand":
            return (rand(3, 1),)
        case "matrix":
            return rand(3, 5), rand(5, 4)
        case "batched_matrix":
            return rand(2, 3, 5), rand(2, 5, 4)
        case "addmm":
            return rand(3, 4), rand(3, 5), rand(5, 4)
        case "vector":
            return rand(5), rand(5)
        case "index_select":
            return rand(3, 4), indices([3, 1, 1, 0])
        case "gather":
            return rand(3, 4), indices([[3, 0], [1, 1], [0, 2]])
        case "scatter":
            return rand(3, 4), indices([[3, 0], [1, 2], [0, 2]]), rand(3, 2)
        case "embedding":
            return indices([[0, 3, 1], [4, 1, 0]]), rand(5, 4)
        case "linear":
            return rand(2, 5), rand(4, 5), rand(4)
        case "conv1d":
            return rand(2, 3, 8), rand(4, 3, 3), rand(4)
        case "conv2d":
            return rand(2, 3, 6, 6), rand(4, 3, 3, 3), rand(4)
        case "image":
            return (rand(2, 3, 6, 6),)
        case "layer_norm":
            return rand(2, 3, 4), rand(4), rand(4)
        case _:
            raise ValueError(f"Unknown input fixture: {kind}")


def operator_configs(
    cuda_device: torch.device, *, low_precision: bool = False
) -> tuple[tuple[torch.device, torch.dtype], ...]:
    """Preserve CPU/CUDA coverage and selected low-precision CUDA cases."""
    configs = [(torch.device("cpu"), torch.float32), (cuda_device, torch.float32)]
    if low_precision:
        configs.extend((cuda_device, dtype) for dtype in (torch.float16, torch.bfloat16))
    return tuple(configs)


def make_operator_check(
    test: unittest.TestCase,
    operation: Callable,
    kind: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[], None]:
    """Own one JIT instance outside the allocation-check callback.

    All input/output tensors are local to the callback. Its first invocation
    checks compilation and fresh-value cache reuse; subsequent invocations
    exercise the same specialization during the CUDA allocation checks.
    """
    compiled = trident.jit(operation, dynamic=False)

    def check() -> None:
        for seed in (17, 29):
            with test.subTest(seed=seed):
                args = make_inputs(kind, device, dtype, seed)
                expected = operation(*args)
                actual = compiled(*args)
                torch.testing.assert_close(actual, expected)
                test.assertEqual(len(compiled._sub_modules), 1)

    return check
