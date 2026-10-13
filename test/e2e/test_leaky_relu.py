# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation leaky_relu correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class LeakyReluTest(unittest.TestCase):
    def test_leaky_relu_eager_reference(self) -> None:
        def leaky_relu(x):
            return F.leaky_relu(x, negative_slope=0.1)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, leaky_relu, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
