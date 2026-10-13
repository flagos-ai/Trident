# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Cached factory results through the public wrapper."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class ArangeTest(unittest.TestCase):
    def test_cached_factory_result(self) -> None:
        @trident.jit
        def create_arange() -> torch.Tensor:
            return torch.arange(0, 64, 2, dtype=torch.float32, device="cuda")

        def check() -> None:
            expected = torch.arange(0, 64, 2, dtype=torch.float32, device="cuda")
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = create_arange()
                    self.assertIsInstance(result, torch.Tensor)
                    torch.testing.assert_close(result, expected)
            self.assertEqual(len(create_arange._sub_modules), 1)
            # This public wrapper boundary is absent from direct FX import tests.
            wrapper_module = f"{create_arange._sub_modules[0]}"
            self.assertIn("tvm_ffi.cast", wrapper_module)
            self.assertIn("-> !tvm_ffi.any", wrapper_module)
            self.assertNotIn("!torch.union<", wrapper_module)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_arange_eager_reference(self) -> None:
        def arange(x):
            return torch.arange(0, 8, 2, device=x.device, dtype=x.dtype)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(self, arange, "unary", device, dtype)
                assert_no_cuda_leak(self, check, cuda_device)
