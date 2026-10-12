# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Transpose view strides and shared storage."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class TransposeTest(unittest.TestCase):
    def test_transpose_view(self) -> None:
        @trident.jit
        def transpose(x: torch.Tensor) -> torch.Tensor:
            return x.t()

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = transpose(x)
                    torch.testing.assert_close(result, x.t())
                    self.assertEqual(result.shape, torch.Size([3, 2]))
                    self.assertEqual(result.stride(), x.t().stride())
                    self.assertEqual(result.data_ptr(), x.data_ptr())
                    x[0, 0] += 1.0
                    self.assertEqual(result[0, 0].item(), x[0, 0].item())
            self.assertEqual(len(transpose._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
