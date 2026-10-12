# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""CUDA cached-execution allocation checks."""

import gc
import unittest
from collections.abc import Callable

import torch


def assert_no_cuda_leak(
    test: unittest.TestCase,
    action: Callable[[], None],
    device: torch.device,
) -> None:
    """Check live allocations after each cached execution in the current process."""
    assert device.type == "cuda" and device.index is not None

    def allocated_bytes() -> int:
        gc.collect()
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)

    for _ in range(2):
        action()
    baseline = allocated_bytes()
    for iteration in range(1, 11):
        action()
        current = allocated_bytes()
        test.assertEqual(
            current,
            baseline,
            f"{test.id()}: CUDA allocation leak on {device}, iteration={iteration}, "
            f"baseline={baseline}, current={current}, delta={current - baseline}",
        )
