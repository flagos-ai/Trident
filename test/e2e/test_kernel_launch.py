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
def increment_kernel(value: tl.tensor, BLOCK: tl.constexpr) -> None:
    offsets = tl.arange(0, BLOCK)
    tl.store(value + offsets, tl.load(value + offsets) + 1)


@triton.jit
def shift_kernel(value: tl.tensor, BLOCK: tl.constexpr, MODE: tl.constexpr) -> None:
    offsets = tl.arange(0, BLOCK)
    delta = 1 if MODE == "increment" else -1
    tl.store(value + offsets, tl.load(value + offsets) + delta)


@triton.jit
def add_runtime_kernel(value: tl.tensor, amount: tl.int64, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(value + offsets, tl.load(value + offsets) + amount)


class KernelLaunchTest(unittest.TestCase):
    def test_specialization_rejection_falls_through_cached_candidate(self) -> None:
        @trident.jit
        def add_runtime(value: torch.Tensor, amount: int) -> None:
            add_runtime_kernel[(1, 1, 1)](value, amount, BLOCK=64)

        value = torch.zeros(64, device="cuda")

        # The first candidate specializes amount=16 as divisible by 16.
        add_runtime(value, 16)
        torch.testing.assert_close(value, torch.full_like(value, 16))
        self.assertEqual(len(add_runtime._sub_modules), 1)

        # amount=8 shares the Dynamo path but rejects the first Triton
        # specialization. Exactly one generic candidate is then compiled.
        value.zero_()
        add_runtime(value, 8)
        torch.testing.assert_close(value, torch.full_like(value, 8))
        self.assertEqual(len(add_runtime._sub_modules), 2)

        def check() -> None:
            # Cached dispatch rejects main_0 and succeeds in main_1 without
            # returning to Python to compile a duplicate specialization.
            value.zero_()
            add_runtime(value, 8)
            torch.testing.assert_close(value, torch.full_like(value, 8))
            self.assertEqual(len(add_runtime._sub_modules), 2)

            # Candidate order remains stable: the aligned candidate still wins.
            value.zero_()
            add_runtime(value, 16)
            torch.testing.assert_close(value, torch.full_like(value, 16))
            self.assertEqual(len(add_runtime._sub_modules), 2)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

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
                    if operation.name == "tvm_ffi_api.kernel_launch":
                        launches.append(operation)
                    return ir.WalkResult.ADVANCE

                with compiled.ctx:
                    module, *_ = compiled._sub_modules
                    module.operation.walk(collect)
                    self.assertEqual(len(launches), 1)
                    [launch] = launches
                    _, *remaining_attributes = launch.attributes["specializations"]
                    attribute = ir.Attribute.parse(
                        "#tvm_ffi_api.variable_specialization<kind = !llvm.ptr, "
                        "divisibility = 9223372036854775808>"
                    )
                    launch.attributes["specializations"] = ir.ArrayAttr.get(
                        [attribute, *remaining_attributes]
                    )
                    compiled.executor = compiled.stub_compile()
                reject_first = False
                value.zero_()

            for phase in ("first", "cached"):
                with self.subTest(phase=phase):
                    value.zero_()
                    compiled(value)
                    # The rejected candidate suppresses its later effects, then
                    # the newly compiled/cached fallback executes exactly once.
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
            module, *_ = increment._sub_modules
            self.assertIn("!tvm_ffi_monad.result<!tvm_ffi_abi.none>", f"{module}")

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
