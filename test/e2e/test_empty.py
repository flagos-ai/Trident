# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""empty allocation metadata and independent storage."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class EmptyTest(unittest.TestCase):
    def test_empty_allocation(self) -> None:
        def empty(x: torch.Tensor) -> torch.Tensor:
            return torch.empty((3, 4), device=x.device, dtype=x.dtype)

        compiled = trident.jit(empty)

        def check() -> None:
            x = torch.zeros((3, 4), device="cuda", dtype=torch.float32)
            # Keep both allocations alive to check allocation independence.
            results: list[torch.Tensor] = []
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = compiled(x)
                    self.assertEqual(result.shape, x.shape)
                    self.assertEqual(result.dtype, x.dtype)
                    self.assertEqual(result.device, x.device)
                    self.assertNotEqual(result.data_ptr(), 0)
                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
                    results.append(result)
            self.assertNotEqual(results[0].data_ptr(), results[1].data_ptr())
            self.assertEqual(len(compiled._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
