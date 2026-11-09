# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Dynamic shapes, scalar arithmetic and control flow through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class DynamicTest(unittest.TestCase):
    def test_cond(self) -> None:
        @trident.jit(dynamic=True)
        def conditional(x: torch.Tensor):
            return torch.cond(x.shape[0] > 4, lambda t: t + 1, lambda t: t - 1, (x,))

        def check() -> None:
            for size in (3, 6, 8):
                x = torch.randn(size, device="cuda")
                expected = x + 1 if size > 4 else x - 1
                for phase in ("warmup", "cached"):
                    with self.subTest(size=size, phase=phase):
                        torch.testing.assert_close(conditional(x), expected)
            self.assertIn("scf.if", str(conditional._sub_modules[0]))

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

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

    def test_scalar_arithmetic(self) -> None:
        def arithmetic(x: torch.Tensor):
            size = x.shape[0]
            return x + 1, (
                size + 2,
                size - 2,
                size * 3,
                size**3,
                size / 2,
                (-size) // 3,
                (-size) % 3,
                size // -3,
                size % -3,
                size << 1,
                size >> 1,
                -(size / 2),
                True + size / 2,
                torch.sym_min(size, 4),
                torch.sym_max(size, 4),
                size == 5,
                size != 5,
                size < 5,
                size <= 5,
                size > 5,
                size >= 5,
                torch.sym_not(size == 5),
                size / 2 + True,
                (size == 5) | True,
            )

        compiled = trident.jit(arithmetic, dynamic=True)

        def check() -> None:
            for size in (3, 5, 7):
                x = torch.randn(size, device="cuda")
                expected = arithmetic(x)
                for phase in ("first", "repeated"):
                    with self.subTest(size=size, phase=phase):
                        tensor, scalars = compiled(x)
                        torch.testing.assert_close(tensor, expected[0])
                        self.assertEqual(scalars, expected[1])
                        self.assertEqual(
                            tuple(map(type, scalars)), tuple(map(type, expected[1]))
                        )
            self.assertEqual(len(compiled._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
