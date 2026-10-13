# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation conv2d correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class Conv2dTest(unittest.TestCase):
    def test_conv2d_eager_reference(self) -> None:
        def conv2d(x, w, b):
            return F.conv2d(x, w, b, padding=1)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, conv2d, "conv2d", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
