# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT
"""Native allocation/view metadata, ownership, and compiled error propagation."""

import gc
import unittest
from pathlib import Path

import torch
import trident
import tvm_ffi
from trident.core import ir, passmanager, register_all_dialects, register_all_passes
from trident.core.execution_engine import ExecutionEngine
from trident.utils.runtime import find_runtime_libraries


def allocate_like(x):
    return torch.empty_like(x)


def allocate_channels_last(x):
    return torch.empty_like(x, dtype=torch.float16, memory_format=torch.channels_last)


def allocate_empty(x):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def allocate_strided(x):
    return torch.empty_strided(x.shape, (1, 2, 6, 24), dtype=x.dtype, device=x.device)


def allocate_zeros(x):
    return torch.zeros(x.shape, dtype=x.dtype, device=x.device)


def allocate_zeros_like(x):
    return torch.zeros_like(x)


def view_noop(x):
    return x.view(x.shape[0], x.shape[1])


def view_reshape(x):
    return x.view(x.shape[0] * x.shape[1], x.shape[2])


class DirectAllocationsTest(unittest.TestCase):
    def test_metadata(self):
        functions = [
            allocate_like,
            allocate_channels_last,
            allocate_empty,
            allocate_strided,
            allocate_zeros,
            allocate_zeros_like,
        ]
        for device in ["cpu", "cuda"]:
            x = torch.empty((2, 3, 4, 5), device=device).transpose(2, 3)
            for function in functions:
                compiled = trident.jit(function)
                expected = function(x)
                for _ in range(3):
                    actual = compiled(x)
                    for attribute in ["shape", "dtype", "device", "layout"]:
                        self.assertEqual(
                            getattr(actual, attribute), getattr(expected, attribute)
                        )
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.is_pinned(), expected.is_pinned())
                    if function in (allocate_zeros, allocate_zeros_like):
                        self.assertTrue(torch.equal(actual, expected))

    def test_view_noop_and_reshape(self):
        for device in ["cpu", "cuda"]:
            x2d = torch.randn((4, 8), device=device)
            c_noop = trident.jit(view_noop)
            for _ in range(3):
                actual = c_noop(x2d)
                self.assertEqual(actual.shape, x2d.shape)
                self.assertTrue(torch.equal(actual, x2d))

            x3d = torch.randn((2, 3, 4), device=device)
            c_reshape = trident.jit(view_reshape)
            for _ in range(3):
                actual = c_reshape(x3d)
                self.assertEqual(actual.shape, (6, 4))
                self.assertTrue(torch.equal(actual, x3d.view(6, 4)))

    def test_lifetime(self):
        compiled = trident.jit(allocate_like)
        x = torch.empty((1024, 128), device="cuda")
        for _ in range(5):
            result = compiled(x)
            del result
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _ in range(1000):
            result = compiled(x)
            del result
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), baseline)

    def test_compiled_error_and_recovery(self):
        # Bypass Dynamo's negative-shape rejection to exercise the compiled
        # shim failure path, including its output status and cleanup.
        fixture = (
            Path(__file__).resolve().parents[2]
            / "core/test/Dialect/TVMFFI/API/direct-allocations.mlir"
        )
        source = fixture.read_text().replace(
            "@allocate()", "@allocate(%n: !tvm_ffi_abi.int)"
        )
        source = source.replace("  %n = tvm_ffi_abi.constant.int 8\n", "", 1)
        with ir.Context() as context:
            register_all_dialects(context)
            register_all_passes()
            module = ir.Module.parse(source)
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)"
            ).run(module.operation)
            engine = ExecutionEngine(
                module, opt_level=3, shared_libs=find_runtime_libraries()
            )
            engine.initialize()
            function = tvm_ffi.Function.__from_mlir_packed_safe_call__(
                engine.raw_lookup("__tvm_ffi_allocate"), keep_alive_object=engine
            )
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "negative"):
                    function(-1)
            result = function(8)
            self.assertEqual(torch.from_dlpack(result).shape, (8, 8))
