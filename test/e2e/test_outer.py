# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation outer correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class OuterTest(unittest.TestCase):
    def test_outer_eager_reference(self) -> None:
        def outer(x, y):
            return torch.outer(x, y)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, outer, "vector", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
