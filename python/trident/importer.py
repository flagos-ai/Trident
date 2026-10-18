# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Import FX graphs and compiled Triton kernels into semantic MLIR.

Scalars stay native while computing and are boxed at FFI boundaries.
ATen call order, tensor aliases, and mutations are preserved.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Sequence
from functools import cached_property, reduce
from typing import Any, TypeAlias, cast

import sympy
import torch
import triton
import tvm_ffi
from torch.export.graph_signature import (
    ConstantArgument,
    InputKind,
    OutputKind,
)
from torch.utils import _pytree as pytree

from trident.core import ir
from trident.core.dialects import arith, cf, func, gpu, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d

from .input import InputPath, InputTable
from .ir import (
    array_type,
    binary,
    bind,
    box,
    boxed_type,
    constant,
    container_create,
    is_binary_operation,
    is_value_in_scope,
    make_result_types,
    native,
    native_constant,
    native_type,
    pure,
    tuple_type,
    value_type,
)

ImportedValue: TypeAlias = (
    ir.Value
    | tuple["ImportedValue", ...]
    | list["ImportedValue"]
    | dict[str, "ImportedValue"]
    | torch.fx.GraphModule
    | None
)


class TridentFxImporter:
    def __init__(self, *, context: ir.Context, specialization_id: int) -> None:
        self.context = context
        self.specialization_id = specialization_id
        with context, ir.Location.unknown(context):
            self.module = ir.Module.create()
        self.values: dict[torch.fx.Node, ImportedValue] = {}
        self.symbols: dict[Any, ir.Value] = {}

    def _aten_argument(
        self, node: torch.fx.Node, index: int, parameter: torch.Argument
    ) -> Any:
        if index < len(node.args):
            argument = node.args[index]
        elif parameter.name in node.kwargs:
            argument = node.kwargs[parameter.name]
        elif parameter.has_default_value():
            argument = parameter.default_value
        else:
            raise ValueError(f"missing argument {parameter.name} for {node.target}")
        if isinstance(argument, (torch.layout, torch.memory_format)):
            argument = self._convert_torch_enum(argument)
        return argument

    def _bind(self, node: torch.fx.Node, value: ImportedValue) -> None:
        self.values[node] = value
        metadata = node.meta.get("val")
        if isinstance(
            metadata, (torch.SymInt, torch.SymFloat, torch.SymBool)
        ) and isinstance(value, ir.Value):
            self.symbols[metadata.node.expr] = native(value)
        if isinstance(metadata, torch.Tensor) and isinstance(value, ir.Value):
            # Bind bare input shape symbols to runtime metadata. Composite
            # expressions are evaluated from their symbols, never from hints.
            for index, size in enumerate(metadata.shape):
                if (
                    isinstance(size, torch.SymInt)
                    and size.node.expr.is_Symbol
                    and size.node.expr not in self.symbols
                ):
                    self.symbols[size.node.expr] = tvm_ffi_d.tensor_size(
                        value, native_constant(index)
                    )

    def _build_aten(self, node: torch.fx.Node) -> ImportedValue:
        name = node.target._schema.name.removeprefix("aten::")
        if name in {
            "_assert_scalar",
            "dim",
            "sym_numel",
            "sym_size",
            "sym_storage_offset",
            "sym_stride",
        }:
            return self._build_aten_special(node, name)
        return self._build_aten_call(node)

    def _build_aten_call(self, node: torch.fx.Node) -> ImportedValue:
        target = node.target
        assert isinstance(target, torch._ops.OpOverload)
        schema = target._schema
        scalar_arguments = [
            index
            for index, argument in enumerate(node.args)
            if index < len(schema.arguments)
            and schema.arguments[index].type.kind() == "TensorType"
            and isinstance(
                argument.meta.get("val")
                if isinstance(argument, torch.fx.Node)
                else argument,
                (bool, int, float, torch.SymInt, torch.SymFloat, torch.SymBool),
            )
        ]
        if scalar_arguments and "Scalar" in target.overloadpacket.overloads():
            candidate = target.overloadpacket.Scalar._schema
            if all(
                index < len(candidate.arguments)
                and candidate.arguments[index].name == schema.arguments[index].name
                and candidate.arguments[index].type.kind()
                in ("NumberType", "IntType", "FloatType", "BoolType")
                for index in scalar_arguments
            ):
                schema = candidate
        arguments = [
            self._aten_argument(node, index, parameter)
            for index, parameter in enumerate(schema.arguments)
        ]
        metadata = node.meta.get("val")
        payload_types = [
            value_type(value)
            for value in ((metadata,) if len(schema.returns) == 1 else metadata or ())
        ]
        result_types = make_result_types(payload_types)
        with bind(
            self.build_arguments(arguments), result_types=result_types
        ) as operation:
            name = schema.name.removeprefix("aten::")
            registry_name = f"trident.aten.{name}" + (
                f".{schema.overload_name}" if schema.overload_name else ""
            )
            call = self.build_call(
                registry_name,
                operation.arguments[: len(arguments)],
                value_type(metadata),
            )
            if len(schema.returns) == 1:
                tvm_ffi_d.MonadYieldOp(call)
            else:
                with bind(call, result_types=result_types) as invocation:
                    [array] = invocation.arguments
                    tvm_ffi_d.MonadYieldOp(self._build_results(array, payload_types))
                tvm_ffi_d.MonadYieldOp(invocation.outputs)
        results = tuple(operation.outputs)
        if len(schema.returns) <= 1:
            [result] = results
            return result
        return results

    def _build_aten_special(self, node: torch.fx.Node, name: str) -> ImportedValue:
        arguments = node.args[:1] if name == "_assert_scalar" else node.args
        values = [self.argument(value) for value in arguments]
        operations = {
            "dim": tvm_ffi_d.tensor_dim,
            "sym_size": tvm_ffi_d.tensor_size,
            "sym_storage_offset": tvm_ffi_d.tensor_storage_offset,
            "sym_stride": tvm_ffi_d.tensor_stride,
        }
        if operation := operations.get(name):
            return operation(*(native(value) for value in values))
        if name == "sym_numel":
            [tensor], [argument] = values, arguments
            return reduce(
                arith.muli,
                (
                    tvm_ffi_d.tensor_size(tensor, native_constant(index))
                    for index in range(argument.meta["val"].dim())
                ),
                native_constant(1),
            )
        assert name == "_assert_scalar"
        [condition] = values
        _, *messages = node.args
        message, *_ = messages or ["FX assertion failed"]
        cf.AssertOp(native(condition), message)
        return constant(None)

    def _build_cond(self, node: torch.fx.Node) -> ImportedValue:
        predicate, true_graph, false_graph, operands = node.args
        branch_types = make_result_types(
            value_type(element) for element in node.meta["val"]
        )
        with bind(
            self.build_arguments([*operands, predicate]),
            result_types=branch_types,
        ) as operation:
            *operand_values, condition = operation.arguments
            conditional = scf.IfOp(native(condition), branch_types, has_else=True)
            for block, graph in (
                (conditional.then_block, true_graph),
                (conditional.else_block, false_graph),
            ):
                child = self.values[graph]
                assert isinstance(child, torch.fx.GraphModule)
                importer = TridentFxImporter(
                    context=self.context,
                    specialization_id=self.specialization_id,
                )
                importer.module = self.module
                with ir.InsertionPoint(block):
                    placeholders = [
                        n for n in child.graph.nodes if n.op == "placeholder"
                    ]
                    for placeholder, operand in zip(
                        placeholders, operand_values, strict=True
                    ):
                        importer._bind(placeholder, operand)
                    output = next(n for n in child.graph.nodes if n.op == "output")
                    (outputs,) = output.args
                    scf.YieldOp(
                        importer.visit(
                            iter(child.graph.nodes),
                            child,
                            outputs,
                            result_types=branch_types,
                        )
                    )
            tvm_ffi_d.MonadYieldOp(conditional.results)
        if node.meta["val"]:
            return tuple(operation.outputs)
        else:
            [result] = operation.outputs
            return result

    def _build_kernel(self, node: torch.fx.Node) -> ImportedValue:
        loc = ir.Location.current
        knodes = cast(dict[str, Any], node.kwargs["kwargs"])
        constant_args_idx: int = node.kwargs["constant_args_idx"]
        constant_args = cast(
            dict[str, Any],
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
                constant_args_idx
            ),
        )
        kernel_idx: int = node.kwargs["kernel_idx"]
        function: triton.KernelInterface = (
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
                kernel_idx
            )
        )
        device = triton.runtime.driver.active.get_current_device()
        configs: list[triton.Config] = getattr(function, "configs", [])
        best_config: triton.Config | None = getattr(function, "best_config", None)
        while not isinstance(function, triton.JITFunction):
            function = function.fn
        kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
        binder_values = {**knodes, **constant_args}
        binder_args = {
            parameter.name: TridentFxImporter._resolve_triton_binder_value(
                parameter.name, binder_values[parameter.name]
            )
            for parameter in function.params
            if parameter.name in binder_values
        }
        config_kwargs = {} if best_config is None else best_config.all_kwargs()
        runtime_kwargs = {
            "debug": binder_args.get("debug", function.debug)
            or triton.knobs.runtime.debug,
            "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
        }
        bound_arguments, specialization, options = binder(
            **binder_args, **config_kwargs, **runtime_kwargs
        )
        key: str = triton.runtime.jit.compute_cache_key(
            kernel_key_cache, specialization, options
        )
        kernel: triton.compiler.CompiledKernel | None = kernel_cache.get(key)
        assert kernel is not None, (
            f"failed to get compiled Triton kernel for {node.name}"
        )
        grids = node.kwargs["grid"]
        if configs and best_config is not None:
            grid = grids[configs.index(best_config)]
        else:
            [grid] = grids
        arguments = cast(dict[str, Any], node.kwargs["kwargs"])
        output_names: list[str] = node.kwargs.get("tensors_to_clone", [])
        specifications: dict[str, tuple[ir.Attribute, Any]] = {}
        for parameter, (triton_type, descriptor), (
            compiled_name,
            compiled_type,
        ) in zip(
            function.params,
            specialization,
            kernel.src.signature.items(),
            strict=True,
        ):
            name = parameter.name
            assert (compiled_name, compiled_type) == (name, triton_type), (
                f"compiled Triton signature does not match its source parameter order: expected {(name, triton_type)!r}, got {(compiled_name, compiled_type)!r}"
            )
            if triton_type == "constexpr":
                constexpr_value = self._constant_value_attribute(descriptor, name)
                specialization_attr = ir.Attribute.parse(
                    f"#tvm_ffi.constant_specialization<value = {constexpr_value}>"
                )
            else:
                if triton_type.startswith("*"):
                    abi_type = "!llvm.ptr"
                elif triton_type in {
                    "i1",
                    "i16",
                    "i32",
                    "i64",
                    "i8",
                    "u1",
                    "u16",
                    "u32",
                    "u64",
                    "u8",
                }:
                    abi_type = f"i{triton_type[1:]}"
                elif triton_type in {"fp32", "fp64"}:
                    abi_type = f"f{triton_type[2:]}"
                else:
                    raise RuntimeError(
                        f"unsupported Triton argument type: {triton_type}"
                    )
                divisibility = ", divisibility = 16" if descriptor == "D" else ""
                specialization_attr = ir.Attribute.parse(
                    f"#tvm_ffi.variable_specialization<kind = {abi_type}{divisibility}>"
                )
            if name in arguments:
                operand = arguments[name]
            elif name in bound_arguments:
                operand = bound_arguments[name]
            else:
                raise RuntimeError(
                    f"missing runtime argument for {name} of type {triton_type}"
                )
            specifications[name] = (specialization_attr, operand)

        result_types = make_result_types(
            tvm_ffi_d.TensorType.get() for _ in output_names
        )
        with bind(
            self.build_arguments([value for _, value in specifications.values()]),
            result_types=result_types,
        ) as operation:
            operands = {
                name: (attribute, box(value))
                for (name, (attribute, _)), value in zip(
                    specifications.items(),
                    operation.arguments[: len(specifications)],
                    strict=True,
                )
            }
            tvm_ffi_d.MonadYieldOp(
                self._build_launch(kernel, node, loc, operands, output_names, grid)
            )
        if output_names:
            return dict(zip(output_names, operation.outputs, strict=True))
        else:
            [result] = operation.outputs
            return result

    def _build_launch(
        self,
        kernel: triton.compiler.CompiledKernel,
        node: torch.fx.Node,
        loc: ir.Location,
        operands: dict[str, tuple[ir.Attribute, ir.Value]],
        names: Sequence[str],
        grid: Sequence[Any],
        index: int = 0,
    ) -> list[ir.Value]:
        result_types = make_result_types(tvm_ffi_d.TensorType.get() for _ in names)
        if index < len(names):
            name = names[index]
            attribute, value = operands[name]
            with bind(
                self.build_call(
                    "trident.aten.clone",
                    [value, constant(None)],
                    tvm_ffi_d.TensorType.get(),
                ),
                result_types=result_types,
            ) as operation:
                [tensor] = operation.arguments
                cloned = {**operands, name: (attribute, tensor)}
                tvm_ffi_d.MonadYieldOp(
                    self._build_launch(
                        kernel, node, loc, cloned, names, grid, index + 1
                    )
                )
            return [*operation.outputs]
        with bind(self.build_arguments(grid), result_types=result_types) as operation:
            cubin = kernel.asm["cubin"]
            binary_name = f"_trident_s{self.specialization_id}_{node.name}"
            module_op = self.module.operation
            module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
            if all(
                not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
                for op in self.module.body.operations
            ):
                cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
                gpu_object = ir.Attribute.parse(
                    f'#gpu.object<#nvvm.target<chip = "sm_{kernel.metadata.target.arch}">, '
                    f'"{cubin_mlir}">'
                )
                with ir.InsertionPoint(self.module.body):
                    gpu.binary(
                        binary_name,
                        ir.ArrayAttr.get([gpu_object]),
                        offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                        loc=loc,
                    )
            tvm_ffi_d.kernel_launch(
                ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
                *(native(value) for value in operation.arguments),
                native_constant(kernel.metadata.num_warps * kernel.metadata.warp_size),
                native_constant(1),
                native_constant(1),
                [value for _, value in operands.values()],
                ir.ArrayAttr.get([attribute for attribute, _ in operands.values()]),
                dynamic_shared_memory_size=arith.constant(
                    ir.IntegerType.get_signless(32), kernel.metadata.shared, loc=loc
                ),
                loc=loc,
            )
            outputs: list[ir.Value] = []
            for name in names:
                _, value = operands[name]
                outputs.append(value)
            tvm_ffi_d.MonadYieldOp(pure(outputs, result_types=result_types))
        return [*operation.outputs]

    def _build_results(
        self, array: ir.Value, types: Sequence[ir.Type]
    ) -> list[ir.Value]:
        result_types = make_result_types(types)

        def build(values: list[ir.Value]) -> list[ir.Value]:
            index = len(values)
            if index == len(types):
                return pure(values, result_types=result_types)
            with bind(
                self.build_container_get(array, index, types[index]),
                result_types=result_types,
            ) as operation:
                tvm_ffi_d.MonadYieldOp(build([*values, *operation.arguments]))
            return [*operation.outputs]

        return build([])

    def _build_scalar(self, node: torch.fx.Node) -> ImportedValue:
        arguments = node.args
        if node.target is torch.sym_sum and len(arguments) == 1:
            [argument] = arguments
            if isinstance(argument, (tuple, list)):
                arguments = argument
        if node.target is operator.pow:
            base, exponent = node.args
            if not isinstance(exponent, int) or exponent < 0:
                raise NotImplementedError(
                    "symbolic integer power requires a nonnegative constant exponent"
                )
            elif exponent == 0:
                return native_constant(1)
            arguments = (base,)
        if not any(
            isinstance(
                self.values[value] if isinstance(value, torch.fx.Node) else value,
                (tuple, list),
            )
            for value in arguments
        ):
            return self._scalar(node, [self.argument(value) for value in arguments])
        result_types = [tvm_ffi_d.ResultType.get(value_type(node.meta["val"]))]
        with bind(
            self.build_arguments(arguments), result_types=result_types
        ) as operation:
            tvm_ffi_d.MonadYieldOp(
                pure(
                    [self._scalar(node, operation.arguments)],
                    result_types=result_types,
                )
            )
        [result] = operation.outputs
        return result

    def _call_function(self, node: torch.fx.Node) -> ImportedValue:
        target = node.target
        if target is operator.getitem:
            aggregate, index = node.args
            imported = self.values[aggregate]
            if not isinstance(imported, ir.Value):
                return imported[index]
            else:
                [result] = self.build_container_get(
                    imported, index, value_type(node.meta["val"])
                )
                return result
        elif isinstance(target, torch._ops.OpOverload):
            return self._build_aten(node)
        elif isinstance(target, torch._ops.HigherOrderOperator):
            if target.__name__ in (
                "triton_kernel_wrapper_functional",
                "triton_kernel_wrapper_mutation",
            ):
                return self._build_kernel(node)
            elif target.__name__ == "cond":
                return self._build_cond(node)
            else:
                raise NotImplementedError(
                    f"unsupported higher-order FX operation: {target}"
                )
        else:
            return self._build_scalar(node)

    @staticmethod
    def _constant_value_attribute(value: Any, name: str) -> ir.Attribute:
        if isinstance(value, tuple):
            return ir.ArrayAttr.get(
                [
                    TridentFxImporter._constant_value_attribute(
                        element, f"{name}[{index}]"
                    )
                    for index, element in enumerate(value)
                ]
            )
        elif isinstance(value, bool):
            return ir.BoolAttr.get(value)
        elif isinstance(value, int):
            assert -(1 << 63) <= value < (1 << 63), (
                f"constexpr specialization {name!r} does not fit in i64: {value}"
            )
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        elif isinstance(value, float):
            return ir.FloatAttr.get_f64(value)
        elif isinstance(value, str):
            return ir.StringAttr.get(value)
        else:
            raise NotImplementedError(
                f"unsupported constexpr specialization {name!r}: {value!r}"
            )

    @cached_property
    def _convert_torch_enum(self) -> tvm_ffi.Function:
        return tvm_ffi.get_global_func("trident.runtime.torch_enum_to_int")

    @staticmethod
    def _integer_power(base: ir.Value, exponent: int) -> ir.Value:
        base = native(base)
        result = native_constant(1)
        for _ in range(exponent):
            result = arith.muli(result, base)
        return result

    @staticmethod
    def _resolve_triton_binder_value(
        name: str,
        value: Any,
    ) -> Any:
        if isinstance(value, torch.fx.Node):
            assert "val" in value.meta, (
                f"Triton binder argument {name} node {value.name} "
                "does not contain metadata value"
            )
            return (
                triton.MockTensor(dtype=metadata_value.dtype)
                if isinstance(metadata_value := value.meta["val"], torch.Tensor)
                else TridentFxImporter._resolve_triton_binder_value(
                    name, metadata_value
                )
            )
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return TridentFxImporter._resolve_triton_binder_value(name, value.node.hint)
        elif isinstance(value, tuple):
            items = tuple(
                TridentFxImporter._resolve_triton_binder_value(name, item)
                for item in value
            )
            return type(value)(*items) if hasattr(value, "_fields") else items
        elif isinstance(value, list):
            return [
                TridentFxImporter._resolve_triton_binder_value(name, item)
                for item in value
            ]
        elif isinstance(value, dict):
            return {
                key: TridentFxImporter._resolve_triton_binder_value(name, item)
                for key, item in value.items()
            }
        else:
            return value

    def _scalar(self, node: torch.fx.Node, values: Sequence[ir.Value]) -> ir.Value:
        target = {torch.sym_max: max, torch.sym_min: min}.get(node.target, node.target)
        if is_binary_operation(target):
            return binary(target, *values)
        elif target is torch.sym_sum:
            return reduce(
                arith.addi, (native(value) for value in values), native_constant(0)
            )
        elif target is operator.pow:
            [base] = values
            _, exponent = node.args
            return self._integer_power(base, exponent)
        elif target in (operator.neg, operator.not_, torch.sym_not):
            (value,) = values
            value = native(value)
            return (
                arith.xori(value, native_constant(True))
                if target is not operator.neg
                else binary(operator.sub, native_constant(0), value)
            )
        else:
            raise NotImplementedError(f"unsupported FX call: {target}")

    def argument(self, value: Any) -> ir.Value:
        """Import a non-container value without opening a failure scope."""
        if isinstance(value, torch.fx.Node):
            value = self.values[value]
        if isinstance(value, ir.Value):
            assert is_value_in_scope(value), (
                "argument payload is outside its defining block"
            )
            return value
        elif isinstance(value, (torch.fx.GraphModule, dict, tuple, list)):
            raise NotImplementedError(
                f"aggregate requires visitor construction: {value}"
            )
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.symbolic(value.node.expr)
        else:
            return constant(value)

    def argument_type(self, value: Any) -> ir.Type:
        if isinstance(value, torch.fx.Node):
            value = self.values[value]
        if isinstance(value, (list, tuple)):
            types = [
                tvm_ffi_d.IntType.get()
                if isinstance(item, torch.dtype)
                else self.argument_type(item)
                for item in (
                    self.values[item] if isinstance(item, torch.fx.Node) else item
                    for item in value
                )
            ]
            return tuple_type(types) if isinstance(value, tuple) else array_type(types)
        return (
            value.type
            if isinstance(value, ir.Value)
            else (
                tvm_ffi_d.RawStrType.get()
                if isinstance(value, str)
                else native_type(value_type(value))
            )
        )

    def build_arguments(self, arguments: Sequence[Any]) -> list[ir.Value]:
        """Construct arguments only; successful payloads never leave their bind."""
        arguments = [
            self.values[value] if isinstance(value, torch.fx.Node) else value
            for value in arguments
        ]
        result_types = make_result_types(
            tvm_ffi_d.IntType.get()
            if isinstance(value, torch.dtype)
            else self.argument_type(value)
            for value in arguments
        )

        def build(start: int, values: list[ir.Value]) -> list[ir.Value]:
            for index in range(start, len(arguments)):
                argument = arguments[index]
                if isinstance(argument, torch.dtype):
                    current = self.build_call(
                        "trident.runtime.tvm_ffi_to_torch_type",
                        [self.argument(argument)],
                        tvm_ffi_d.IntType.get(),
                    )
                elif isinstance(argument, (tuple, list)):
                    type_ = self.argument_type(argument)
                    element_types = [tvm_ffi_d.ResultType.get(type_)]
                    with bind(
                        self.build_arguments(argument), result_types=element_types
                    ) as elements:
                        tvm_ffi_d.MonadYieldOp(
                            container_create(elements.arguments[: len(argument)], type_)
                        )
                    current = elements.outputs
                else:
                    values.append(self.argument(argument))
                    continue
                with bind(current, result_types=result_types) as operation:
                    tvm_ffi_d.MonadYieldOp(
                        build(index + 1, [*values, *operation.arguments])
                    )
                return [*operation.outputs]
            return pure(values, result_types=result_types)

        return build(0, [])

    def build_call(
        self, name: str, arguments: Sequence[ir.Value], result: ir.Type
    ) -> list[ir.Value]:
        """Return an asserting invocation wrapped as a successful Result."""
        assert all(is_value_in_scope(value) for value in arguments), (
            "call argument payload is outside its defining block"
        )
        handle = tvm_ffi_d.FunctionGetGlobalOp(name)
        call = tvm_ffi_d.FunctionCallOp(
            result, handle.result, [box(value) for value in arguments]
        )
        return pure([call.result])

    def build_container_get(
        self, container: ir.Value, index: int | ir.Value, target: ir.Type
    ) -> list[ir.Value]:
        type_ = container.type
        if isinstance(type_, tvm_ffi_d.TupleType):
            assert isinstance(index, int)
            assert -len(type_.types) <= index < len(type_.types), (
                "tuple index out of range"
            )
            payload = type_.types[index]
            values = tvm_ffi_d.TupleUnconstructOp(type_.types, container).elements
            [result] = pure([values[index]])
        elif isinstance(type_, tvm_ffi_d.ArrayType):
            payload = type_.element_type
            [result] = pure(
                [
                    tvm_ffi_d.ArrayGetOp(
                        container,
                        native_constant(index) if isinstance(index, int) else index,
                    ).result
                ]
            )
        else:
            return self.build_call(
                "ffi.ArrayGetItem", [container, native_constant(index)], target
            )
        if payload == target:
            return [result]
        result_types = [tvm_ffi_d.ResultType.get(target)]
        with bind([result], result_types=result_types) as extracted:
            [value] = extracted.arguments
            boxed = boxed_type(target)
            if value.type != boxed:
                if isinstance(
                    boxed, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
                ) and not isinstance(
                    value.type, (tvm_ffi_d.AnyType, tvm_ffi_d.UnionType)
                ):
                    value = tvm_ffi_d.upcast(boxed, value)
                else:
                    value = tvm_ffi_d.downcast(boxed, value)
            if target != boxed:
                value = native(value)
            tvm_ffi_d.MonadYieldOp(pure([value], result_types=result_types))
        return [*extracted.outputs]

    def build_input(self, table: InputTable, path: InputPath) -> list[ir.Value]:
        """Extract one structural path, caching only complete Results."""
        node = table.node(path)
        assert node is not None, f"input cannot be resolved: {path}"
        if node.value is not None:
            return pure(
                [node.value], result_types=[tvm_ffi_d.ResultType.get(node.type)]
            )
        elif node.result is not None and all(
            is_value_in_scope(value) for value in node.result
        ):
            return node.result
        else:
            assert len(path) > 1, f"input cannot be resolved: {path}"
            types = [tvm_ffi_d.ResultType.get(node.type)]
            *parent_path, index = path
            with bind(
                self.build_input(table, parent_path), result_types=types
            ) as operation:
                [array] = operation.arguments
                tvm_ffi_d.MonadYieldOp(
                    self.build_container_get(array, index, node.type)
                )
            node.result = [*operation.outputs]
            return node.result

    def build_inputs(
        self, table: InputTable, paths: Sequence[InputPath] | None = None
    ) -> list[ir.Value]:
        paths = table.input_paths() if paths is None else paths
        types = make_result_types(table.node(path).type for path in paths)

        def build(start: int, values: list[ir.Value]) -> list[ir.Value]:
            for index in range(start, len(paths)):
                path = paths[index]
                value = table[path]
                if value is not None:
                    values.append(value)
                    continue
                with bind(
                    self.build_input(table, path), result_types=types
                ) as operation:
                    tvm_ffi_d.MonadYieldOp(
                        build(index + 1, [*values, *operation.arguments])
                    )
                return [*operation.outputs]
            return pure(values, result_types=types)

        return build(0, [])

    def import_program(
        self, program: torch.export.ExportedProgram, *, func_name: str
    ) -> func.FuncOp:
        producers = {node.name: node for node in program.graph.nodes}
        inputs = [
            spec
            for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
            and not isinstance(spec.arg, ConstantArgument)
        ]
        outputs = [
            spec
            for spec in program.graph_signature.output_specs
            if spec.kind == OutputKind.USER_OUTPUT
        ]
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(self.module.body),
        ):
            input_types = [
                value_type(producers[spec.arg.name].meta["val"]) for spec in inputs
            ]
            payload_types = [
                value_type(
                    spec.arg.value
                    if isinstance(spec.arg, ConstantArgument)
                    else producers[spec.arg.name].meta["val"]
                )
                for spec in outputs
            ]
            function = func.FuncOp(
                func_name,
                ir.FunctionType.get(
                    input_types,
                    make_result_types(payload_types),
                ),
                visibility="private",
            )
            block = function.add_entry_block()
            result_types = function.type.results
            with ir.InsertionPoint(block):
                output_values = [
                    spec.arg.value
                    if isinstance(spec.arg, ConstantArgument)
                    else producers[spec.arg.name]
                    for spec in outputs
                ]
                block_arguments = block.arguments
                initial_values: list[Any] = []
                user_arguments = iter(block_arguments)
                for spec in program.graph_signature.input_specs:
                    if (
                        not isinstance(spec.arg, ConstantArgument)
                        and spec.kind == InputKind.USER_INPUT
                    ):
                        initial_values.append(next(user_arguments))
                    else:
                        state = (
                            program.state_dict
                            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                            else program.constants
                        )
                        initial_values.append(
                            spec.arg.value
                            if isinstance(spec.arg, ConstantArgument)
                            else state[spec.target]
                        )
                with bind(
                    self.build_arguments(initial_values),
                    result_types=result_types,
                ) as initialization:
                    for spec, value in zip(
                        program.graph_signature.input_specs,
                        initialization.arguments[: len(initial_values)],
                        strict=True,
                    ):
                        self._bind(producers[spec.arg.name], value)
                    tvm_ffi_d.MonadYieldOp(
                        self.visit(
                            iter(program.graph.nodes),
                            program.graph_module,
                            output_values,
                            result_types=result_types,
                        )
                    )
                results = [*initialization.outputs]
                func.ReturnOp(results)
        return function

    def symbolic(self, expression: Any) -> ir.Value:
        operations = {
            sympy.Add: operator.add,
            sympy.Max: max,
            sympy.Min: min,
            sympy.Mul: operator.mul,
        }
        if expression in self.symbols:
            return self.symbols[expression]
        elif expression.is_Integer:
            return native_constant(operator.index(expression))
        elif expression.is_Float:
            return native_constant(float(expression))
        elif expression.func in operations:
            return reduce(
                lambda lhs, rhs: binary(operations[expression.func], lhs, rhs),
                (self.symbolic(argument) for argument in expression.args),
            )
        elif expression.func.__name__ in ("FloorDiv", "Mod", "PythonMod"):
            lhs, rhs = (self.symbolic(argument) for argument in expression.args)
            operation = (
                operator.floordiv
                if expression.func.__name__ == "FloorDiv"
                else operator.mod
            )
            return binary(operation, lhs, rhs)
        elif expression.func is sympy.Pow:
            base, exponent = expression.args
            if exponent.is_Integer and exponent >= 0:
                return self._integer_power(
                    self.symbolic(base), operator.index(exponent)
                )
        raise NotImplementedError(
            f"unbound symbolic FX expression: {expression}; refusing to use a concrete hint"
        )

    def visit(
        self,
        nodes: Iterator[torch.fx.Node],
        graph_module: torch.fx.GraphModule,
        outputs: Sequence[Any],
        *,
        result_types: Sequence[ir.Type],
    ) -> list[ir.Value]:
        """Bind Result leaves and consume the shared iterator in their success scope."""
        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op == "get_attr":
                value = graph_module
                for attribute in f"{node.target}".split("."):
                    value = getattr(value, attribute)
                if isinstance(value, torch.fx.GraphModule):
                    pass
                elif isinstance(value, (tuple, list)):
                    [value] = self.build_arguments([value])
                else:
                    value = self.argument(value)
            elif node.op == "call_function":
                value = self._call_function(node)
            else:
                raise NotImplementedError(f"unsupported FX node: {node.format_node()}")
            leaves, structure = pytree.tree_flatten(value)
            results = {
                index: leaf
                for index, leaf in enumerate(leaves)
                if isinstance(leaf, ir.Value)
                and isinstance(leaf.type, tvm_ffi_d.ResultType)
            }
            if not results:
                self._bind(node, value)
                continue
            with bind([*results.values()], result_types=result_types) as operation:
                for index, argument in zip(results, operation.arguments, strict=True):
                    leaves[index] = (
                        None
                        if isinstance(argument.type, tvm_ffi_d.NoneType)
                        else argument
                    )
                self._bind(node, pytree.tree_unflatten(leaves, structure))
                tvm_ffi_d.MonadYieldOp(
                    self.visit(nodes, graph_module, outputs, result_types=result_types)
                )
            return [*operation.outputs]
        with bind(
            self.build_arguments(outputs), result_types=result_types
        ) as operation:
            tvm_ffi_d.MonadYieldOp(pure(operation.arguments, result_types=result_types))
        return [*operation.outputs]
