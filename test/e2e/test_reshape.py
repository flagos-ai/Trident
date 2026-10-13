# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Dynamic reshape expressions reuse compiled specializations."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class ReshapeTest(unittest.TestCase):
    def test_dynamic_shapes(self) -> None:
        @trident.jit(dynamic=True)
        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0] // 2, x.shape[1] * 2)

        def check() -> None:
            for size in (6, 10, 14):
                x = torch.randn(size, 3, device="cuda")
                for phase in ("first", "repeated"):
                    with self.subTest(size=size, phase=phase):
                        torch.testing.assert_close(reshape(x), x.reshape(size // 2, 6))
            text = str(reshape._sub_modules[0])
            self.assertIn("tvm_ffi.tensor.size", text)
            self.assertIn("arith.floordivsi", text)
            self.assertIn("arith.muli", text)
            self.assertNotIn("!torch.", text)
            self.assertEqual(len(reshape._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_reshape_eager_reference(self) -> None:
        def reshape(x):
            return torch.reshape(x, (2, 6))

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, reshape, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
