# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""ATen overloads, schema arguments and multiple results through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class OperatorTest(unittest.TestCase):
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

    def test_arithmetic_overloads(self) -> None:
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

    def test_multiple_aten_results(self) -> None:
        @trident.jit
        def topk(x: torch.Tensor):
            return torch.topk(x, 2)

        def check() -> None:
            x = torch.tensor([3.0, -1.0, 2.0, 0.0], device="cuda")
            expected = torch.topk(x, 2)
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    values, indices = topk(x)
                    torch.testing.assert_close(values, expected.values)
                    torch.testing.assert_close(indices, expected.indices)
            self.assertEqual(len(topk._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
