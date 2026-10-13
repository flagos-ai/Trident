# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation mm correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class MmTest(unittest.TestCase):
    def test_mm_eager_reference(self) -> None:
        def mm(x, y):
            return torch.mm(x, y)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, mm, "matrix", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
