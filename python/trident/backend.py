# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import gc
import inspect
import operator
from collections.abc import Callable, Hashable
from itertools import islice
from typing import Any, Final

import torch
import tvm_ffi
import tvm_ffi.utils
from torch.export._trace import _extract_fake_inputs
from torch.export.graph_signature import (
    InputKind,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)
from torch.utils import _pytree as pytree

from trident import capi_utils
from trident.core import (
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from trident.core.dialects import transform
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.execution_engine import ExecutionEngine
from trident.ffi import Exception
from trident.ir import array_type, value_type

from . import pytree as _pytree  # noqa: F401 - Register scalar nodes before export.
from .guards import parse_guards
from .guards.scope import Scope
from .guards.tree import Root
from .importer import TridentFxImporter


def _convert_input(value: Any) -> Any:
    if isinstance(value, torch.device):
        return tvm_ffi.device(f"{value}")
    elif isinstance(value, torch.dtype):
        return tvm_ffi.convert(value)
    else:
        return value


def _convert_output(value: Any) -> Any:
    return torch.from_dlpack(value) if isinstance(value, tvm_ffi.Tensor) else value


def _map_containers(value: Any, convert: Callable[[Any], Any]) -> Any:
    """Convert leaves while preserving Python containers and unpacking FFI Arrays."""
    if isinstance(value, (tvm_ffi.Array, list)):
        return [_map_containers(element, convert) for element in value]
    elif isinstance(value, tuple):
        return tuple(_map_containers(element, convert) for element in value)
    elif isinstance(value, dict):
        return {
            key: _map_containers(element, convert) for key, element in value.items()
        }
    else:
        return convert(value)


class TridentGraphModule:
    """Imports a Torch FX graph into MLIR and executes through TVM FFI.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus a TVM FFI
    dispatcher. The first candidate whose guards match executes exactly once;
    its result or failure ends the call.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.executor(*args, **kwargs)
        if isinstance(result, Exception):
            kind = result.get_kind()
            if kind == "GuardMiss":
                return self.compile(*args, **kwargs)
            raise RuntimeError(f"Compiled execution failed: {kind}")
        return result

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        dynamic: bool = True,
    ) -> None:
        self._signature = inspect.signature(fn)
        self.fn: Final[Callable[..., Any]] = fn
        self.dynamic: Final[bool] = dynamic
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: list[ir.Module] = []
        self._dispatcher = Root()
        self.executor: Callable[..., Any] = self.compile

    def __name__(self) -> str:
        return self.fn.__name__

    def _register_shape(
        self,
        value: Any,
        fake_value: Any,
        shape_env: Any,
        dimensions: dict[Hashable, torch.export.Dim],
        dynamic_shapes: torch.export.ShapesCollection,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            return
        assert isinstance(fake_value, torch.Tensor), (
            "Dynamo placeholder for a Tensor input does not contain FakeTensor metadata"
        )
        assert value.dim() == fake_value.dim(), (
            f"Dynamo FakeTensor has rank {fake_value.dim()}, but its input "
            f"has rank {value.dim()}"
        )
        shape: dict[int, torch.export.Dim] = {}
        for index, size in enumerate(fake_value.shape):
            if not isinstance(size, torch.SymInt):
                continue
            expression = size.node.expr
            symbols = expression.free_symbols
            concrete_size = value.shape[index]
            if not symbols:
                assert expression.is_Integer is True, (
                    f"Dynamo dimension {expression} is neither static nor symbolic"
                )
                assert operator.index(round(expression)) == concrete_size, (
                    f"Dynamo dimension {expression} does not match input size "
                    f"{concrete_size}"
                )
                continue
            assert len(symbols) == 1, (
                f"cannot represent Dynamo dimension {expression} with torch.export.Dim"
            )
            [symbol] = symbols
            coefficient = expression.coeff(symbol)
            offset = expression.subs(symbol, 0)
            assert (
                coefficient.is_Integer is True
                and coefficient > 0
                and offset.is_Integer is True
                and expression == coefficient * symbol + offset
            ), f"cannot represent Dynamo dimension {expression} with torch.export.Dim"
            dimension = dimensions.get(symbol)
            if dimension is None:
                value_range = shape_env.var_to_range.get(symbol)
                options: dict[str, int] = {}
                if value_range is not None:
                    if value_range.lower.is_Integer is True:
                        options["min"] = operator.index(round(value_range.lower))
                    if value_range.upper.is_Integer is True:
                        options["max"] = operator.index(round(value_range.upper))
                dimension = torch.export.Dim.AUTO(**options)
                dimensions[symbol] = dimension
            if coefficient != 1:
                dimension = operator.index(round(coefficient)) * dimension
            if offset != 0:
                dimension = dimension + operator.index(round(offset))
            shape[index] = dimension
        dynamic_shapes[value] = shape

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Export and import one specialization, then rebuild its dispatcher."""
        index = len(self._sub_modules)

        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        exported_gm, gs = torch._dynamo.export(
            self.fn,
            aten_graph=True,
            assume_static_by_default=not self.dynamic,
        )(*args, **kwargs)
        fake_args, fake_kwargs, fake_mode = _extract_fake_inputs(
            exported_gm, args, kwargs
        )
        shape_env = fake_mode.shape_env
        assert shape_env is not None, "Dynamo FakeTensorMode does not have a ShapeEnv"
        dimensions: dict[Hashable, torch.export.Dim] = {}
        dynamic_shapes = torch.export.ShapesCollection()

        pytree.tree_map(
            lambda value, fake_value: self._register_shape(
                value, fake_value, shape_env, dimensions, dynamic_shapes
            ),
            (args, kwargs),
            (fake_args, fake_kwargs),
        )
        exported_program: torch.export.ExportedProgram = torch.export.export(
            copy.deepcopy(exported_gm),
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        warmup_result = torch.fx.Interpreter(exported_program.module()).run(
            *pytree.tree_leaves(exported_program.example_inputs)
        )

        self._output_spec = pytree.tree_structure(warmup_result)

        importer = TridentFxImporter(context=self.ctx, specialization_id=index)
        main_func_name = f"main_{index}"

        # Describe original parameters and execution-local FX input bindings.
        with self.ctx:
            input_specs = exported_program.graph_signature.input_specs
            exported_input_values = pytree.tree_leaves(exported_program.example_inputs)
            assert len(input_specs) == len(exported_input_values), (
                f"ExportedProgram input specs do not match flattened exported inputs: got {len(input_specs)} specs and {len(exported_input_values)} values"
            )

            exported_args, exported_kwargs = exported_program.example_inputs
            exported_bound = self._signature.bind(*exported_args, **exported_kwargs)
            positional_names = [
                name
                for name, parameter in self._signature.parameters.items()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            assert len(exported_args) <= len(positional_names), (
                "variadic positional inputs are unsupported"
            )
            # Traverse in export order; dispatch parameters retain signature order.
            provided_names = [
                *islice(positional_names, len(exported_args)),
                *exported_kwargs,
            ]
            input_spec_iter = iter(input_specs)
            producers = {node.name: node for node in exported_program.graph.nodes}
            input_paths: dict[str, tuple[int | str, ...]] = {}

            def build_type(value: Any, input_path: tuple[int | str, ...]) -> ir.Type:
                if isinstance(value, (torch.device, torch.dtype)):
                    return value_type(value)
                if isinstance(value, (list, tuple)):
                    child_types = [
                        build_type(child, (*input_path, index))
                        for index, child in enumerate(value)
                    ]
                    return (
                        tvm_ffi_d.TupleType.get(child_types)
                        if isinstance(value, tuple)
                        else array_type(child_types)
                    )
                assert pytree.tree_structure(value).is_leaf(), (
                    f"unsupported input container: {type(value)}"
                )
                input_spec = next(input_spec_iter)
                if input_spec.kind == InputKind.USER_INPUT and isinstance(
                    input_spec.arg,
                    (SymBoolArgument, SymFloatArgument, SymIntArgument, TensorArgument),
                ):
                    input_paths[input_spec.arg.name] = input_path
                    return value_type(producers[input_spec.arg.name].meta["val"])
                return value_type(value)

            input_types = {
                name: build_type(exported_bound.arguments[name], (name,))
                for name in provided_names
            }
            input_types = {
                name: input_types[name] if name in input_types else value_type(value)
                for name, value in bound.arguments.items()
            }
        importer.import_program(
            exported_program,
            func_name=main_func_name,
            input_types=input_types,
            input_paths=input_paths,
        )
        torch._dynamo.reset()
        self._dispatcher.add(parse_guards(gs), input_types, main_func_name)
        self._sub_modules.append(importer.module)
        self.executor = self.stub_compile()
        gc.collect()
        return warmup_result

    def stub_compile(self) -> Callable[..., Any]:
        """Build the combined module and semantic dispatcher, lower it,
        JIT-compile, and return a callable executor.

        Called after each export to include the new specialization.
        """
        # Merge cloned sub-modules so repeated compilation preserves the originals.
        fn_file: str = inspect.getfile(self.fn)
        _, fn_line = inspect.getsourcelines(self.fn)
        combined: ir.Module = ir.Module.create(
            loc=ir.Location.file(fn_file, fn_line, col=0, context=self.ctx)
        )
        for sub_mod in self._sub_modules:
            with self.ctx, sub_mod.operation.location:
                for attr_name in sub_mod.operation.attributes:
                    if (
                        attr_name not in combined.operation.attributes
                        and not attr_name.startswith("sym_name")
                    ):
                        combined.operation.attributes[attr_name] = (
                            sub_mod.operation.attributes[attr_name]
                        )

                transform.interpreter.copy_symbols_and_merge_into(
                    combined.operation,
                    sub_mod.operation,
                )

        self._build_dispatcher(combined)

        # Lower all functions, including the dispatcher, together.
        with self.ctx:
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)",
            ).run(combined.operation)

        # Keep the execution engine alive through the FFI callable.
        engine: ExecutionEngine = ExecutionEngine(
            combined,
            opt_level=3,
            shared_libs=capi_utils.find_runtime_libraries(),
        )
        engine.initialize()

        symbol: str = f"__tvm_ffi_{self.fn.__name__}"
        ptr: int = engine.raw_lookup(symbol)
        assert ptr is not None, (
            f"symbol not found: {symbol}; "
            f"available: {[op.sym_name.value for op in combined.body.operations if hasattr(op, 'sym_name')]}"
        )

        fn: tvm_ffi.Function = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr,
            keep_alive_object=engine,
        )

        def invoke(*values: Any) -> Any:
            result = _map_containers(
                fn(*(_map_containers(value, _convert_input) for value in values)),
                _convert_output,
            )

            if isinstance(result, Exception):
                kind = result.get_kind()
                if kind != "GuardMiss":
                    raise RuntimeError(f"Compiled execution failed: {kind}")
                return result
            count = self._output_spec.num_leaves
            leaves = [] if count == 0 else [result] if count == 1 else list(result)
            return pytree.tree_unflatten(leaves, self._output_spec)

        return tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
            invoke, self._signature
        )

    def _build_dispatcher(self, combined: ir.Module) -> None:
        """Generate the ordered guard tree directly as an execute-region CFG."""
        with self.ctx, combined.operation.location, ir.InsertionPoint(combined.body):
            any_type = tvm_ffi_d.AnyType.get()
            parameter_types = [tvm_ffi_d.PackArgumentsType.get()]
            dispatcher = tvm_ffi_d.PackFuncOp(
                self.fn.__name__,
                ir.TypeAttr.get(ir.FunctionType.get(parameter_types, [any_type])),
            )
            entry = dispatcher.add_entry_block()
            with ir.InsertionPoint(entry):
                [arguments] = entry.arguments
                success, payload = self._dispatcher.build(Scope(arguments))
                tvm_ffi_d.return_(success, payload)
