# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation conv1d correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class Conv1dTest(unittest.TestCase):
    def test_conv1d_eager_reference(self) -> None:
        def conv1d(x, w, b):
            return F.conv1d(x, w, b, padding=1)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, conv1d, "conv1d", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
