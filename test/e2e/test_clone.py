# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Clone memory formats and independent storage."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class CloneTest(unittest.TestCase):
    def test_clone_memory_formats(self) -> None:
        def contiguous(x: torch.Tensor) -> torch.Tensor:
            return x.clone(memory_format=torch.contiguous_format)

        def preserve(x: torch.Tensor) -> torch.Tensor:
            return x.clone(memory_format=torch.preserve_format)

        cases = (
            ("contiguous", contiguous, torch.contiguous_format),
            ("preserve", preserve, torch.preserve_format),
        )
        compiled_cases = tuple(
            (name, trident.jit(function), memory_format)
            for name, function, memory_format in cases
        )

        def check() -> None:
            for name, compiled, memory_format in compiled_cases:
                x = (
                    torch.arange(64, device="cuda", dtype=torch.float32)
                    .reshape(2, 32)
                    .t()
                    .clone(memory_format=torch.preserve_format)
                )
                assert not x.is_contiguous()
                for phase in ("warmup", "cached"):
                    with self.subTest(memory_format=name, phase=phase):
                        expected = x.clone(memory_format=memory_format)
                        result = compiled(x)
                        torch.testing.assert_close(result, expected)
                        self.assertEqual(result.stride(), expected.stride())
                        self.assertEqual(
                            result.is_contiguous(), expected.is_contiguous()
                        )
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        x[0, 0] += 1.0
                        torch.testing.assert_close(result, expected)
                self.assertEqual(len(compiled._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
