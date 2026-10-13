# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Multi-result top-k values and indices through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class TopkTest(unittest.TestCase):
    def test_multiple_aten_results(self) -> None:
        @trident.jit
        def topk(x: torch.Tensor):
            return torch.topk(x, 2)

        def check() -> None:
            x = torch.tensor([3.0, -1.0, 2.0, 0.0], device="cuda")
            expected = torch.topk(x, 2)
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    values, indices = topk(x)
                    torch.testing.assert_close(values, expected.values)
                    torch.testing.assert_close(indices, expected.indices)
            self.assertEqual(len(topk._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_topk_eager_reference(self) -> None:
        def topk(x):
            return torch.topk(x, k=2, dim=1)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, topk, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
