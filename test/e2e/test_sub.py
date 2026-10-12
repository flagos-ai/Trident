# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Scalar and tensor subtraction overloads through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class SubTest(unittest.TestCase):
    def test_scalar_and_tensor_overloads(self) -> None:
        def sub_scalar(x: torch.Tensor) -> torch.Tensor:
            return torch.sub(x, 0.5, alpha=2.0)

        def sub_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.sub(x, y, alpha=1.0)

        compiled_scalar = trident.jit(sub_scalar)
        compiled_tensor = trident.jit(sub_tensor)

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            y = torch.full_like(x, 2.0)
            for name, compiled, arguments, expected in (
                ("sub_scalar", compiled_scalar, (x,), x - 1.0),
                ("sub_tensor", compiled_tensor, (x, y), x - y),
            ):
                for phase in ("warmup", "cached"):
                    with self.subTest(operation=name, phase=phase):
                        torch.testing.assert_close(compiled(*arguments), expected)
                self.assertEqual(len(compiled._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
