# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Scalar multiplication through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


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

    def test_mul_eager_reference(self) -> None:
        def mul(x, y):
            return torch.mul(x, y)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, mul, "broadcast", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)

    def test_mul_scalar_eager_reference(self) -> None:
        def mul_scalar(x):
            return torch.mul(x, -0.5)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, mul_scalar, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
