# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Single-operation layer_norm correctness and cached-execution allocation checks."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from test.support.cuda import assert_no_cuda_leak
from test.support.operators import make_operator_check, operator_configs


class LayerNormTest(unittest.TestCase):
    def test_layer_norm_eager_reference(self) -> None:
        def layer_norm(x, w, b):
            return F.layer_norm(x, (4,), w, b)

        cuda_device = torch.device("cuda", torch.cuda.current_device())
        for device, dtype in operator_configs(cuda_device, low_precision=True):
            with self.subTest(device=str(device), dtype=dtype):
                check = make_operator_check(
                    self, layer_norm, "layer_norm", device, dtype
                )
                assert_no_cuda_leak(self, check, cuda_device)
