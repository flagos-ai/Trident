# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation add correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class AddTest(unittest.TestCase):
    def test_add_eager_reference(self) -> None:
        def add(x, y):
            return torch.add(x, y)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, add, "binary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)

    def test_add_broadcast_eager_reference(self) -> None:
        def add_broadcast(x, y):
            return torch.add(x, y)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(
                    self, add_broadcast, "broadcast", device, dtype
                )
                assert_no_cuda_leak(self, check, cuda_device)

    def test_add_scalar_eager_reference(self) -> None:
        def add_scalar(x):
            return torch.add(x, 1.5)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, add_scalar, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
