# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Container inputs and output pytree reconstruction."""

from __future__ import annotations

import unittest

import torch
import trident

from test.support.cuda import assert_no_cuda_leak


class ContainerTest(unittest.TestCase):
    def test_container_inputs_unpack(self) -> None:
        @trident.jit
        def list_add(x: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
            return x + values[0] + values[1]

        @trident.jit
        def mixed_add(x: torch.Tensor, values: list) -> torch.Tensor:
            return x + values[0] + values[2]

        @trident.jit
        def nested_add(
            x: torch.Tensor,
            values: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> torch.Tensor:
            return x + values[0][0] + values[1][1]

        @trident.jit
        def tuple_add(
            x: torch.Tensor,
            values: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            return x + values[0] + values[-1]

        def check() -> None:
            x = torch.randn(4, device="cuda")
            first = torch.randn(4, device="cuda")
            second = torch.randn(4, device="cuda")
            for name, function, values, expected in (
                ("list", list_add, [first, second], x + first + second),
                ("mixed", mixed_add, [first, None, 3], x + first + 3),
                (
                    "nested",
                    nested_add,
                    [(first, torch.zeros_like(x)), (torch.zeros_like(x), second)],
                    x + first + second,
                ),
                ("tuple", tuple_add, (first, second), x + first + second),
            ):
                for phase in ("warmup", "cached"):
                    with self.subTest(container=name, phase=phase):
                        torch.testing.assert_close(function(x, values), expected)
                self.assertEqual(len(function._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_empty_containers_and_none(self) -> None:
        @trident.jit
        def outputs(x: torch.Tensor):
            scalar = torch.empty([], device=x.device).fill_(2)
            return x + scalar, [], (), None, None

        def check() -> None:
            x = torch.ones(4, device="cuda")
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result, empty_list, empty_tuple, first, second = outputs(x)
                    torch.testing.assert_close(result, x + 2)
                    self.assertEqual(empty_list, [])
                    self.assertEqual(empty_tuple, ())
                    self.assertIsNone(first)
                    self.assertIsNone(second)
            self.assertEqual(len(outputs._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_nested_outputs(self) -> None:
        @trident.jit
        def outputs(x: torch.Tensor):
            return {"tensor": x + 1, "other": (x * 2, [3, None])}

        def check() -> None:
            x = torch.randn(4, device="cuda")
            for phase in ("warmup", "cached"):
                with self.subTest(phase=phase):
                    result = outputs(x)
                    self.assertIsInstance(result, dict)
                    self.assertIsInstance(result["other"], tuple)
                    self.assertEqual(result["other"][1], [3, None])
                    torch.testing.assert_close(result["tensor"], x + 1)
                    torch.testing.assert_close(result["other"][0], x * 2)
            self.assertEqual(len(outputs._sub_modules), 1)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)

    def test_tuple_length_specialization(self) -> None:
        @trident.jit
        def add_length(x: torch.Tensor, values: tuple) -> torch.Tensor:
            return x + len(values)

        def check() -> None:
            x = torch.ones(4, device="cuda")
            for values in ((), (x,), (x, x), (), (x,)):
                for phase in ("warmup", "cached"):
                    with self.subTest(length=len(values), phase=phase):
                        torch.testing.assert_close(
                            add_length(x, values), x + len(values)
                        )
            self.assertEqual(len(add_length._sub_modules), 3)

        device = torch.device("cuda", torch.cuda.current_device())
        assert_no_cuda_leak(self, check, device)
