# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Device, dtype and string specialization through public JIT."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class SpecializationTest(unittest.TestCase):
    def test_argument_specialization(self) -> None:
        @trident.jit
        def empty_on_device(
            x: torch.Tensor,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.empty_like(x, device=device).zero_()

        @trident.jit
        def empty_with_dtype(
            x: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.empty_like(x, dtype=dtype).zero_()

        def check() -> None:
            x: torch.Tensor = torch.randn(4, device="cuda")
            # Cache reuse needs an explicit ordinal, matching the runtime device.
            device: torch.device = x.device
            first: torch.Tensor = empty_on_device(x, device)
            repeated: torch.Tensor = empty_on_device(x, device)
            imported_module = f"{empty_on_device._sub_modules[0]}"
            self.assertIn("trident.aten.empty_like", imported_module)
            self.assertNotIn("trident.aten.empty_permuted", imported_module)
            torch.testing.assert_close(first, torch.zeros_like(x))
            torch.testing.assert_close(repeated, torch.zeros_like(x))

            for phase in ("warmup", "cached"):
                with self.subTest(device="cpu", phase=phase):
                    cpu_result: torch.Tensor = empty_on_device(x, torch.device("cpu"))
                    self.assertEqual(cpu_result.device, torch.device("cpu"))
                    torch.testing.assert_close(cpu_result, torch.zeros_like(cpu_result))
            self.assertEqual(len(empty_on_device._sub_modules), 2)

            for dtype in (torch.float32, torch.float16, torch.float32):
                for phase in ("first", "repeated"):
                    with self.subTest(dtype=dtype, phase=phase):
                        result: torch.Tensor = empty_with_dtype(x, dtype)
                        self.assertEqual(result.dtype, dtype)
                        torch.testing.assert_close(result, torch.zeros_like(result))
            self.assertEqual(len(empty_with_dtype._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

        x = torch.randn(4, device=device)
        # Preserve the unindexed CUDA spelling's execution coverage separately.
        implicit_result = empty_on_device(x, torch.device("cuda"))
        torch.testing.assert_close(implicit_result, torch.zeros_like(x))

    def test_string_specialization(self) -> None:
        @trident.jit
        def named(x: torch.Tensor, name: str):
            return name, x + 1

        def check() -> None:
            x = torch.randn(4, device="cuda")
            for name in ("x", "a string longer than the inline ABI storage"):
                first = named(x, name)
                repeated = named(x, name)
                self.assertEqual(first[0], name)
                self.assertEqual(repeated[0], name)
                torch.testing.assert_close(first[1], x + 1)
                torch.testing.assert_close(repeated[1], x + 1)
            self.assertEqual(len(named._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
