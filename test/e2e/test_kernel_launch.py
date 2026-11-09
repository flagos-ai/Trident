# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Mutation-only Triton launches return a None payload."""

import unittest

import torch
import trident
import triton
import triton.language as tl
from trident.core import ir

from test.support.cuda import assert_no_cuda_leak


@triton.jit
def increment_kernel(value: tl.tensor, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(value + offsets, tl.load(value + offsets) + 1)


@triton.jit
def shift_kernel(value: tl.tensor, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    delta = 1 if MODE == "increment" else -1
    tl.store(value + offsets, tl.load(value + offsets) + delta)


class KernelLaunchTest(unittest.TestCase):
    def test_cleanup_after_rejection(self) -> None:
        @trident.jit
        def compiled(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            temporary = value.clone()
            increment_kernel[(1, 1, 1)](temporary, BLOCK=64)
            value.add_(1)
            return temporary, temporary

        reject_first = True

        def check() -> None:
            nonlocal reject_first
            value = torch.zeros(64, device="cuda")
            if reject_first:
                # Build the first specialization, then make its launch reject every
                # non-null user-space pointer. This exercises rejection after clone,
                # independent of Dynamo's earlier argument guards.
                compiled(value)
                launches: list[ir.Operation] = []

                def collect(operation: ir.Operation) -> ir.WalkResult:
                    if operation.name == "tvm_ffi.kernel_launch":
                        launches.append(operation)
                    return ir.WalkResult.ADVANCE

                with compiled.ctx:
                    compiled._sub_modules[0].operation.walk(collect)
                    self.assertEqual(len(launches), 1)
                    attributes = list(launches[0].attributes["specializations"])
                    attributes[0] = ir.Attribute.parse(
                        "#tvm_ffi.variable_specialization<kind = !llvm.ptr, "
                        "divisibility = 9223372036854775808>"
                    )
                    launches[0].attributes["specializations"] = ir.ArrayAttr.get(
                        attributes
                    )
                    compiled.executor = compiled.stub_compile()
                reject_first = False
                value.zero_()

            for phase in ("first", "cached"):
                with self.subTest(phase=phase):
                    value.zero_()
                    first, second = compiled(value)
                    torch.testing.assert_close(first, torch.ones_like(value))
                    torch.testing.assert_close(second, first)
                    self.assertEqual(first.data_ptr(), second.data_ptr())
                    # Rejection must suppress the in-place effect before retry.
                    torch.testing.assert_close(value, torch.ones_like(value))
            self.assertEqual(len(compiled._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

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

    def test_string_specialization(self) -> None:
        @trident.jit
        def shift(value: torch.Tensor, mode: str) -> None:
            shift_kernel[(1, 1, 1)](value, BLOCK=64, MODE=mode)

        def check() -> None:
            for mode, delta in (("increment", 1), ("decrement", -1)):
                for phase in ("first", "cached"):
                    with self.subTest(mode=mode, phase=phase):
                        value = torch.zeros(64, device="cuda")
                        self.assertIsNone(shift(value, mode))
                        torch.testing.assert_close(value, torch.full_like(value, delta))
            self.assertEqual(len(shift._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
