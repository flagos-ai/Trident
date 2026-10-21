# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, Final, Self

import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import (
    InputKind,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import (
    array_type,
    bind,
    container_get,
    make_result_types,
    pure,
    tuple_type,
    value_type,
)
from trident.ownership import is_value_in_scope

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


class InputNodeBuilder:
    """Static recipe for recursively building table-owned input nodes."""

    __slots__ = ("children", "type")

    def __init__(
        self,
        type: ir.Type,
        children: Sequence[InputNodeBuilder] | None = None,
    ) -> None:
        self.children: Final[list[InputNodeBuilder] | None] = (
            None if children is None else [*children]
        )
        self.type: Final[ir.Type] = type

    def build(self, value: ir.Value | None = None) -> _InputNode:
        """Recursively build table-owned nodes with an optional root value."""
        return _InputNode(
            self.type,
            None
            if self.children is None
            else [child.build() for child in self.children],
            value,
        )


class InputTable:
    """Table-owned input nodes addressable through Python pytree paths."""

    def __getitem__(self, path: Sequence[int | str]) -> ir.Value | None:
        """Read a root or consumer-local value without extracting an input."""
        node = self.node(path)
        return None if node is None else node.value

    def __init__(
        self, nodes: Mapping[str, _InputNode], input_names: Sequence[str]
    ) -> None:
        self._nodes: Final[dict[str, _InputNode]] = {**nodes}
        self._input_names: Final[list[str]] = [*input_names]
        assert all(name in self._nodes for name in self._input_names)

    def clear_results(self) -> None:
        def clear(node: _InputNode) -> None:
            node.result = None
            for child in node.children or ():
                clear(child)

        for node in self._nodes.values():
            clear(node)

    def extract(self, path: Sequence[int | str]) -> list[ir.Value]:
        """Extract one structural path, caching only complete Results."""
        node = self.node(path)
        assert node is not None, f"input cannot be resolved: {path}"
        if node.value is not None:
            return pure(
                [node.value], result_types=[tvm_ffi_d.ResultType.get(node.type)]
            )
        elif node.result is not None and all(
            is_value_in_scope(value) for value in node.result
        ):
            # The cache borrows a reference until block exit; return an independent one.
            with bind(
                node.result, result_types=[value.type for value in node.result]
            ) as cached:
                tvm_ffi_d.MonadYieldOp(pure(cached.arguments))
            return [*cached.outputs]
        else:
            assert len(path) > 1, f"input cannot be resolved: {path}"
            types = [tvm_ffi_d.ResultType.get(node.type)]
            *parent_path, index = path
            with bind(self.extract(parent_path), result_types=types) as operation:
                [array] = operation.arguments
                tvm_ffi_d.MonadYieldOp(container_get(array, index, node.type))

            node.result = [*operation.outputs]
            return node.result

    def extract_many(
        self, paths: Sequence[Sequence[int | str]] | None = None
    ) -> list[ir.Value]:
        paths = self.input_paths() if paths is None else paths
        types = make_result_types(self.node(path).type for path in paths)

        def build(start: int, values: list[ir.Value]) -> list[ir.Value]:
            for index in range(start, len(paths)):
                path = paths[index]
                value = self[path]
                if value is not None:
                    values.append(value)
                    continue
                with bind(self.extract(path), result_types=types) as operation:
                    tvm_ffi_d.MonadYieldOp(
                        build(index + 1, [*values, *operation.arguments])
                    )

                return [*operation.outputs]
            return pure(values, result_types=types)

        return build(0, [])

    def input_paths(self) -> list[tuple[int | str, ...]]:
        """Return leaf paths in exported-input order without emitting IR."""

        def leaves(
            node: _InputNode, path: tuple[int | str, ...]
        ) -> list[tuple[int | str, ...]]:
            if node.children is None:
                return [path]
            else:
                return [
                    leaf
                    for index, child in enumerate(node.children)
                    for leaf in leaves(child, (*path, index))
                ]

        return [
            path
            for name in self._input_names
            for path in leaves(self._nodes[name], (name,))
        ]

    def node(self, path: Sequence[int | str]) -> _InputNode | None:
        assert path, "input paths must start with a parameter name"
        name, *indices = path
        assert isinstance(name, str), "input paths must start with a parameter name"
        node = self._nodes.get(name)
        for index in indices:
            if (
                node is None
                or node.children is None
                or not isinstance(index, int)
                or not 0 <= index < len(node.children)
            ):
                return None
            node = node.children[index]
        return node

    def with_value(self, path: Sequence[int | str], value: ir.Value) -> InputTable:
        """Copy table structure for one lexical consumer without changing shared state."""

        def copy(node: _InputNode) -> _InputNode:
            return _InputNode(
                node.type,
                None
                if node.children is None
                else [copy(child) for child in node.children],
                node.value,
            )

        table = InputTable(
            {name: copy(node) for name, node in self._nodes.items()}, self._input_names
        )
        node = table.node(path)
        assert node is not None
        node.value = value
        return table


class InputTableBuilder:
    """Build the wrapper input schema and bind region-local IR operands."""

    def __init__(
        self,
        entries: Sequence[tuple[str, InputNodeBuilder]],
        input_names: Sequence[str],
    ) -> None:
        self._entries: Final[list[tuple[str, InputNodeBuilder]]] = [*entries]
        self._input_names: Final[list[str]] = [*input_names]
        self.input_types: Final[list[ir.Type]] = [
            builder.type for _, builder in self._entries
        ]

    def build(self, operands: Sequence[ir.Value]) -> InputTable:
        """Bind *operands* to a fresh table for the current IR region."""
        assert len(operands) == len(self._entries), (
            f"input operand count does not match the wrapper signature: got {len(operands)}, expected {len(self._entries)}"
        )
        nodes = {
            name: builder.build(operand)
            for (name, builder), operand in zip(self._entries, operands)
        }
        return InputTable(nodes, self._input_names)

    @classmethod
    def get(
        cls,
        exported_program: torch.export.ExportedProgram,
        signature: inspect.Signature,
        bound_arguments: Mapping[str, Any],
        main_input_types: Sequence[ir.Type],
    ) -> Self:
        input_specs = exported_program.graph_signature.input_specs
        exported_input_values = pytree.tree_leaves(exported_program.example_inputs)
        assert len(input_specs) == len(exported_input_values), (
            f"ExportedProgram input specs do not match flattened exported inputs: got {len(input_specs)} specs and {len(exported_input_values)} values"
        )

        main_input_count = sum(
            input_spec.kind == InputKind.USER_INPUT
            and isinstance(
                input_spec.arg,
                (TensorArgument, SymBoolArgument, SymFloatArgument, SymIntArgument),
            )
            for input_spec in input_specs
        )
        assert main_input_count == len(main_input_types), (
            f"ExportedProgram graph inputs do not match imported MLIR inputs: got {main_input_count} graph inputs and {len(main_input_types)} MLIR inputs"
        )

        in_spec = exported_program.call_spec.in_spec
        assert isinstance(in_spec, pytree.TreeSpec), (
            f"trident.jit requires a pytree TreeSpec for exported inputs; got {in_spec!r}"
        )
        root_children = in_spec.children()
        assert in_spec.type is tuple and len(root_children) == 2, (
            f"unexpected input spec root (expected tuple TreeSpec, got {in_spec})"
        )
        [args_spec, kwargs_spec] = root_children
        assert isinstance(args_spec, pytree.TreeSpec), (
            f"expected args tree spec, got {type(args_spec)!r}"
        )
        assert isinstance(kwargs_spec, pytree.TreeSpec), (
            f"expected kwargs tree spec, got {type(kwargs_spec)!r}"
        )

        signature_names = [*signature.parameters]
        positional_names = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        args_children = args_spec.children()
        kwargs_names = [*kwargs_spec.context]
        kwargs_children = kwargs_spec.children()
        assert len(kwargs_names) == len(kwargs_children), (
            f"ExportedProgram keyword inputs do not match the call spec: got {len(kwargs_children)} inputs and {len(kwargs_names)} names"
        )
        assert len(args_children) <= len(positional_names), (
            f"ExportedProgram positional inputs exceed the positional parameters: got {len(args_children)} inputs and {len(positional_names)} parameters"
        )
        args_names = [name for name, _ in zip(positional_names, args_children)]
        provided_names = [*args_names, *kwargs_names]
        assert len(provided_names) == len(set(provided_names)), (
            f"ExportedProgram input trees map multiple inputs to the same function parameter: {provided_names!r}"
        )
        assert all(name in signature_names for name in provided_names), (
            f"ExportedProgram input trees contain unknown function parameters: got {provided_names!r}, expected names from {signature_names!r}"
        )
        leaf_iter = zip(input_specs, exported_input_values)
        main_input_type_iter = iter(main_input_types)

        def build_node(node: pytree.TreeSpec, name: str) -> InputNodeBuilder:
            if (
                node.type in (torch.device, torch.dtype)
                and not node.children()
                and isinstance(value := node.context, (torch.device, torch.dtype))
            ):
                return InputNodeBuilder(value_type(value), children=[])
            elif node.is_leaf():
                input_spec, value = next(leaf_iter)
                type_ = (
                    next(main_input_type_iter)
                    if input_spec.kind == InputKind.USER_INPUT
                    and isinstance(
                        input_spec.arg,
                        (
                            TensorArgument,
                            SymBoolArgument,
                            SymFloatArgument,
                            SymIntArgument,
                        ),
                    )
                    else value_type(value)
                )
                return InputNodeBuilder(type_)
            else:
                assert node.type in (list, tuple), (
                    f"unsupported input container at {name!r}: {node.type}"
                )
                children = [build_node(child, name) for child in node.children()]
                types = [child.type for child in children]
                return InputNodeBuilder(
                    tuple_type(types) if node.type is tuple else array_type(types),
                    children=children,
                )

        provided_entries = {
            name: build_node(child, name)
            for name, child in chain(
                zip(args_names, args_children), zip(kwargs_names, kwargs_children)
            )
        }

        def build_value(value: Any) -> InputNodeBuilder:
            if isinstance(value, (list, tuple)):
                children = [build_value(child) for child in value]
                types = [child.type for child in children]
                return InputNodeBuilder(
                    tuple_type(types)
                    if isinstance(value, tuple)
                    else array_type(types),
                    children=children,
                )
            else:
                return InputNodeBuilder(value_type(value))

        return cls(
            [
                (
                    name,
                    provided_entries[name]
                    if name in provided_entries
                    else build_value(bound_arguments[name]),
                )
                for name in signature_names
            ],
            provided_names,
        )


class _InputNode:
    """Static input structure and a payload valid only in its visitor scope."""

    __slots__ = ("_value", "children", "result", "type")

    def __init__(
        self, type: ir.Type, children: list[_InputNode] | None, value: ir.Value | None
    ) -> None:
        self.children: Final[list[_InputNode] | None] = children
        self.type: Final[ir.Type] = type
        self._value = value
        self.result: list[ir.Value] | None = None

    @property
    def value(self) -> ir.Value | None:
        assert self._value is None or is_value_in_scope(self._value), (
            "input payload is outside its defining block"
        )
        return self._value

    @value.setter
    def value(self, value: ir.Value | None) -> None:
        self._value = value
