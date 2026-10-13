# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation silu correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class SiluTest(unittest.TestCase):
    def test_silu_eager_reference(self) -> None:
        def silu(x):
            return F.silu(x)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, silu, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
