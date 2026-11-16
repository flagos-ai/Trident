# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Torch scalar pytree registrations required before export."""

import torch
import torch.utils._pytree as pytree

pytree.register_pytree_node(
    torch.device,
    lambda value: ([], value),
    lambda _, value: value,
    flatten_with_keys_fn=lambda value: ([], value),
    serialized_type_name="trident.torch.device",
    to_dumpable_context=str,
    from_dumpable_context=torch.device,
)
pytree.register_pytree_node(
    torch.dtype,
    lambda value: ([], value),
    lambda _, value: value,
    flatten_with_keys_fn=lambda value: ([], value),
    serialized_type_name="trident.torch.dtype",
    to_dumpable_context=str,
    from_dumpable_context=lambda value: getattr(torch, value.removeprefix("torch.")),
)
