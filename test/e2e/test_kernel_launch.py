# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Mutation-only Triton launches return a None payload."""

import unittest

import torch
import trident
import triton
import triton.language as tl

from test.support.cuda import assert_no_cuda_leak


@triton.jit
def increment_kernel(value: tl.tensor, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(value + offsets, tl.load(value + offsets) + 1)


class KernelLaunchTest(unittest.TestCase):
    def test_no_output(self) -> None:
        @trident.jit
        def increment(value: torch.Tensor) -> None:
            increment_kernel[(1, 1, 1)](value, BLOCK=64)

        def check() -> None:
            for initial in (0.0, 2.0):
                for phase in ("warmup", "cached"):
                    with self.subTest(initial=initial, phase=phase):
                        value = torch.full((64,), initial, device="cuda")
                        self.assertIsNone(increment(value))
                        torch.testing.assert_close(
                            value, torch.full_like(value, initial + 1)
                        )
            self.assertEqual(len(increment._sub_modules), 1)
            self.assertIn(
                "!tvm_ffi.monad.result<!tvm_ffi.none>",
                f"{increment._sub_modules[0]}",
            )

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
