# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""In-place addition writeback through an aliased view."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class AddTest(unittest.TestCase):
    def test_inplace_alias(self) -> None:
        @trident.jit
        def increment(x: torch.Tensor) -> torch.Tensor:
            view = x.view(-1)
            x.add_(1)
            return view

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            original = x.clone()
            for increment_count, phase in enumerate(("warmup", "cached"), start=1):
                with self.subTest(phase=phase):
                    expected = original + increment_count
                    result = increment(x)
                    torch.testing.assert_close(x, expected)
                    torch.testing.assert_close(result, expected.flatten())
                    self.assertEqual(result.data_ptr(), x.data_ptr())
            self.assertEqual(len(increment._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
