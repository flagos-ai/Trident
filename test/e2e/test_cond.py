# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Runtime predicates select both branches through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class CondTest(unittest.TestCase):
    def test_cond(self) -> None:
        @trident.jit(dynamic=True)
        def conditional(x: torch.Tensor):
            return torch.cond(x.shape[0] > 4, lambda t: t + 1, lambda t: t - 1, (x,))

        def check() -> None:
            for size in (3, 6, 8):
                x = torch.randn(size, device="cuda")
                expected = x + 1 if size > 4 else x - 1
                for phase in ("warmup", "cached"):
                    with self.subTest(size=size, phase=phase):
                        torch.testing.assert_close(conditional(x), expected)
            self.assertIn("scf.if", str(conditional._sub_modules[0]))

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
