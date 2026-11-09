# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Fresh registered contexts for frontend IR tests."""

from trident.core import ir, register_all_dialects


def new_context() -> ir.Context:
    context = ir.Context()
    register_all_dialects(context)
    return context
