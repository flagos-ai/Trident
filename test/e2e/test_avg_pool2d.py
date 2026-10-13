# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation avg_pool2d correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class AvgPool2dTest(unittest.TestCase):
    def test_avg_pool2d_eager_reference(self) -> None:
        def avg_pool2d(x):
            return F.avg_pool2d(x, kernel_size=2)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, avg_pool2d, "image", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
