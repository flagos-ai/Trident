# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""In-place copy writeback and result aliases."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class CopyTest(unittest.TestCase):
    def test_copy_writeback(self) -> None:
        @trident.jit
        def copy(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
            return destination.copy_(source)

        def check() -> None:
            source = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            destination = torch.zeros_like(source)
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = copy(source, destination)
                    torch.testing.assert_close(destination, source)
                    torch.testing.assert_close(result, source)
                    self.assertEqual(result.data_ptr(), destination.data_ptr())
                    source.add_(1)
            self.assertEqual(len(copy._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
