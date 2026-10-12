# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Scalar multiplication through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class MulTest(unittest.TestCase):
    def test_scalar_overload(self) -> None:
        @trident.jit
        def mul_scalar(x: torch.Tensor) -> torch.Tensor:
            return x * 0.5

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    torch.testing.assert_close(mul_scalar(x), x * 0.5)
            self.assertEqual(len(mul_scalar._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
