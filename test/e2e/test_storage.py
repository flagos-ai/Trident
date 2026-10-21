# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Storage formats, aliases and mutation writeback through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class StorageTest(unittest.TestCase):
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

    def test_copy_writeback(self) -> None:
        @trident.jit
        def copy(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
            return destination.copy_(source)

        def check() -> None:
            source = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            destination = torch.zeros_like(source)
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = copy(source, destination)
                    torch.testing.assert_close(destination, source)
                    torch.testing.assert_close(result, source)
                    self.assertEqual(result.data_ptr(), destination.data_ptr())
                    source.add_(1)
            self.assertEqual(len(copy._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_inplace_alias(self) -> None:
        @trident.jit
        def increment(x: torch.Tensor) -> torch.Tensor:
            view = x.view(-1)
            x.add_(1)
            return view

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            original = x.clone()
            for increment_count, phase in enumerate(("warmup", "cached"), start=1):
                with self.subTest(phase=phase):
                    expected = original + increment_count
                    result = increment(x)
                    torch.testing.assert_close(x, expected)
                    torch.testing.assert_close(result, expected.flatten())
                    self.assertEqual(result.data_ptr(), x.data_ptr())
            self.assertEqual(len(increment._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_transpose_view(self) -> None:
        @trident.jit
        def transpose(x: torch.Tensor) -> torch.Tensor:
            return x.t()

        def check() -> None:
            x = (
                torch.arange(6, device="cuda", dtype=torch.float32)
                .reshape(2, 3)
                .clone()
            )
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = transpose(x)
                    torch.testing.assert_close(result, x.t())
                    self.assertEqual(result.shape, torch.Size([3, 2]))
                    self.assertEqual(result.stride(), x.t().stride())
                    self.assertEqual(result.data_ptr(), x.data_ptr())
                    x[0, 0] += 1.0
                    self.assertEqual(result[0, 0].item(), x[0, 0].item())
            self.assertEqual(len(transpose._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
