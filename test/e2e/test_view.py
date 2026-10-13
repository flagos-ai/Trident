# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation view correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class ViewTest(unittest.TestCase):
    def test_view_eager_reference(self) -> None:
        def view(x):
            return x.view(2, 6)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, view, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
