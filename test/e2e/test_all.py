# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""List-valued reduction dimensions through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class AllTest(unittest.TestCase):
    def test_all_with_list_dims(self) -> None:
        @trident.jit
        def all_over_list_dims(x: torch.Tensor) -> torch.Tensor:
            return torch.all(x, dim=[1, 0], keepdim=False)

        def check() -> None:
            for input_device in (torch.device("cpu"), device):
                x = torch.ones((7, 4, 11, 1), dtype=torch.float32, device=input_device)
                x[0, 0, 0, 0] = 0
                expected = torch.all(x, dim=[1, 0], keepdim=False)
                for phase in ("warmup", "cached"):
                    with self.subTest(device=input_device, phase=phase):
                        result = all_over_list_dims(x)
                        torch.testing.assert_close(result, expected)
                        self.assertEqual(result.shape, torch.Size((11, 1)))
                        self.assertEqual(result.dtype, torch.bool)
            self.assertEqual(len(all_over_list_dims._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_all_eager_reference(self) -> None:
        def all(x):
            return torch.all(x, dim=1)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, all, "boolean_unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
