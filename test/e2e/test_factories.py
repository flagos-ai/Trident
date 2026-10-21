# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Factory metadata and independent allocations through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class FactoryTest(unittest.TestCase):
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

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_empty_allocations(self) -> None:
        def empty(x: torch.Tensor) -> torch.Tensor:
            return torch.empty(x.shape, device=x.device, dtype=x.dtype)

        def empty_like(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        compiled_cases = (
            ("empty", trident.jit(empty)),
            ("empty_like", trident.jit(empty_like)),
        )

        def check() -> None:
            x = torch.zeros((3, 4), device="cuda", dtype=torch.float32)
            for name, compiled in compiled_cases:
                # Keep both results alive to check allocation independence.
                results: list[torch.Tensor] = []
                for phase in ("first", "repeated"):
                    with self.subTest(factory=name, phase=phase):
                        result = compiled(x)
                        self.assertEqual(
                            (result.shape, result.dtype, result.device),
                            (x.shape, x.dtype, x.device),
                        )
                        self.assertNotEqual(result.data_ptr(), 0)
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        results.append(result)
                self.assertNotEqual(results[0].data_ptr(), results[1].data_ptr())
                self.assertEqual(len(compiled._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
