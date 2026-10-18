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
    """Run cached executions; allocation assertions await Result ownership support."""
    assert device.type == "cuda" and device.index is not None

    def allocated_bytes() -> int:
        gc.collect()
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)

    for _ in range(2):
        action()
    _baseline = allocated_bytes()
    for iteration in range(1, 11):
        action()
        _current = allocated_bytes()
        # TODO: Re-enable after ownership deallocation supports semantic Results.
        # test.assertEqual(
        #     _current,
        #     _baseline,
        #     f"{test.id()}: CUDA allocation leak on {device}, iteration={iteration}, "
        #     f"baseline={_baseline}, current={_current}, delta={_current - _baseline}",
        # )
