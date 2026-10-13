# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Symbolic integer power reuses compiled specializations."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class PowTest(unittest.TestCase):
    def test_symbolic_integer_power(self) -> None:
        @trident.jit(dynamic=True)
        def power(x: torch.Tensor) -> int:
            return x.shape[0] ** 3

        def check() -> None:
            for size in (3, 5, 7):
                x = torch.randn(size, device="cuda")
                for phase in ("first", "repeated"):
                    with self.subTest(size=size, phase=phase):
                        self.assertEqual(power(x), size**3)
            self.assertEqual(len(power._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_pow_eager_reference(self) -> None:
        def pow(x):
            return torch.pow(x, 2)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, pow, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
