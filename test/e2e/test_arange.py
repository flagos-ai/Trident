# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Cached factory results through the public wrapper."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


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
            self.assertIn("tvm_ffi.upcast", wrapper_module)
            self.assertIn("-> !tvm_ffi.any", wrapper_module)
            self.assertNotIn("!torch.union<", wrapper_module)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
