# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import gc
import inspect
import operator
from collections.abc import Callable, Hashable, Sequence
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
from trident.core.dialects import (
    scf,
    transform,
)
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.execution_engine import ExecutionEngine
from trident.ffi import Exception

from .guards import parse_guards
from .importer import TridentFxImporter
from .input import InputTableBuilder
from .ir import BINARY_ARITHMETIC, BindScope, native, pure


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
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order - the first successful non-Exception payload wins.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.executor(*args, **kwargs)
        if isinstance(result, Exception):
            # The dispatcher returned an Exception ObjectRef
            # (all specializations failed); compile a new one.
            return self.compile(*args, **kwargs)
        else:
            return result

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        dynamic: bool = True,
    ) -> None:
        super().__init__()
        self.fn: Final[Callable[..., Any]] = fn
        self.dynamic: Final[bool] = dynamic
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: list[ir.Module] = []
        self.executor: Callable[..., Any] = self.stub_compile()

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
        if isinstance(value, torch.Tensor):
            assert isinstance(fake_value, torch.Tensor), (
                "Dynamo placeholder for a Tensor input does not contain "
                "FakeTensor metadata"
            )
            assert value.dim() == fake_value.dim(), (
                f"Dynamo FakeTensor has rank {fake_value.dim()}, but its input "
                f"has rank {value.dim()}"
            )
            shape: dict[int, torch.export.Dim] = {}
            for index, size in enumerate(fake_value.shape):
                if isinstance(size, torch.SymInt):
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
                    else:
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
                        ), (
                            f"cannot represent Dynamo dimension {expression} with torch.export.Dim"
                        )
                        dimension = dimensions.get(symbol)
                        if dimension is None:
                            value_range = shape_env.var_to_range.get(symbol)
                            options: dict[str, int] = {}
                            if value_range is not None:
                                if value_range.lower.is_Integer is True:
                                    options["min"] = operator.index(
                                        round(value_range.lower)
                                    )
                                if value_range.upper.is_Integer is True:
                                    options["max"] = operator.index(
                                        round(value_range.upper)
                                    )
                            dimension = torch.export.Dim.AUTO(**options)
                            dimensions[symbol] = dimension
                        if coefficient != 1:
                            dimension = operator.index(round(coefficient)) * dimension
                        if offset != 0:
                            dimension = dimension + operator.index(round(offset))
                        shape[index] = dimension
            dynamic_shapes[value] = shape

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Export, import and wrap one specialization, then rebuild its dispatcher."""
        index = len(self._sub_modules)

        signature: inspect.Signature = inspect.signature(self.fn)
        trace_args: tuple[Any, ...] = tuple(args)
        bound: inspect.BoundArguments = signature.bind(*trace_args, **kwargs)
        bound.apply_defaults()
        exported_gm, gs = torch._dynamo.export(
            self.fn,
            aten_graph=True,
            assume_static_by_default=not self.dynamic,
        )(*trace_args, **kwargs)
        fake_args, fake_kwargs, fake_mode = _extract_fake_inputs(
            exported_gm, trace_args, kwargs
        )
        shape_env = fake_mode.shape_env
        assert shape_env is not None, "Dynamo FakeTensorMode does not have a ShapeEnv"
        dimensions: dict[Hashable, torch.export.Dim] = {}
        dynamic_shapes = torch.export.ShapesCollection()

        pytree.tree_map(
            lambda value, fake_value: self._register_shape(
                value, fake_value, shape_env, dimensions, dynamic_shapes
            ),
            (trace_args, kwargs),
            (fake_args, fake_kwargs),
        )
        trace_gm = copy.deepcopy(exported_gm)

        exported_program: torch.export.ExportedProgram = torch.export.export(
            trace_gm,
            trace_args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        warmup_result = torch.fx.Interpreter(exported_program.module()).run(
            *pytree.tree_leaves(exported_program.example_inputs)
        )

        self._output_spec = pytree.tree_structure(warmup_result)

        # Step 2: Import FX -> MLIR  ----------------------------------------
        importer: TridentFxImporter = TridentFxImporter(
            context=self.ctx, specialization_id=index
        )
        main_func_name: Final[str] = f"main_{index}"
        main_func: tvm_ffi_d.FuncOp = importer.import_program(
            exported_program, func_name=main_func_name
        )
        with self.ctx:
            main_func.attributes["sym_visibility"] = ir.StringAttr.get("private")
        module: ir.Module = importer.module

        torch._dynamo.reset()

        # Step 3: Wrap with tvm_ffi.func.
        flat_types: Sequence[ir.Type] = main_func.function_type.value.inputs

        with self.ctx:
            input_builder = InputTableBuilder.get(
                exported_program,
                signature,
                bound.arguments,
                flat_types,
            )
        param_types = input_builder.input_types

        with self.ctx:
            abi_result_type = tvm_ffi_d.AnyType.get(context=self.ctx)
            ffi_result_type = tvm_ffi_d.ResultType.get(abi_result_type)
            ffi_type: ir.FunctionType = ir.FunctionType.get(
                param_types, [abi_result_type]
            )

        tvm_ffi_name: Final[str] = f"{self.fn.__name__}_{index}"
        with ir.InsertionPoint(module.body), main_func.operation.location:
            ffi_func: tvm_ffi_d.FuncOp = tvm_ffi_d.FuncOp(
                tvm_ffi_name,
                ir.TypeAttr.get(ffi_type),
                emit_tvm_ffi_abi=ir.UnitAttr.get(self.ctx),
            )
            entry_block: ir.Block = ffi_func.add_entry_block()

            with ir.InsertionPoint(entry_block):
                exception = tvm_ffi_d.exception("GuardMatch")
                rejection = tvm_ffi_d.upcast(abi_result_type, exception)
                with BindScope(
                    [ffi_result_type],
                    parse_guards(gs).build(
                        input_builder, entry_block.arguments, self.ctx
                    ),
                ) as guards:
                    guards_arguments = guards.arguments
                    [predicate] = guards_arguments
                    branch = scf.IfOp(
                        native(predicate), [ffi_result_type], has_else=True
                    )

                    with ir.InsertionPoint(branch.then_block):
                        with BindScope(
                            [ffi_result_type],
                            input_builder.build(entry_block.arguments).extract_many(),
                        ) as inputs:
                            inputs_arguments = inputs.arguments
                            input_specs = exported_program.graph_signature.input_specs
                            input_values = inputs_arguments[: len(input_specs)]
                            assert len(input_values) == len(input_specs), (
                                "unexpected number of reconstructed exported inputs"
                            )
                            main_inputs = [
                                value
                                for spec, value in zip(
                                    input_specs, input_values, strict=True
                                )
                                if spec.kind == InputKind.USER_INPUT
                                and isinstance(
                                    spec.arg,
                                    (
                                        TensorArgument,
                                        SymBoolArgument,
                                        SymFloatArgument,
                                        SymIntArgument,
                                    ),
                                )
                            ]
                            assert len(main_inputs) == len(flat_types), (
                                "unexpected number of reconstructed MLIR inputs"
                            )
                            graph_call = tvm_ffi_d.monad_call(
                                ffi_result_type,
                                ir.FlatSymbolRefAttr.get(main_func_name),
                                main_inputs,
                            )
                            yielded = [graph_call]
                            inputs.yield_(yielded)
                            inputs_results = inputs.results
                        scf.yield_(inputs_results)
                    with ir.InsertionPoint(branch.else_block):
                        yielded = pure([rejection], result_types=[ffi_result_type])
                        scf.yield_(yielded)
                    guards.yield_(branch.results)
                    guards_results = guards.results
                success, payload = tvm_ffi_d.monad_unpack(guards_results[0])
                tvm_ffi_d.return_(tvm_ffi_d.to(success), payload)
        self._sub_modules.append(module)
        self.executor = self.stub_compile()
        gc.collect()
        return warmup_result

    def stub_compile(self) -> Callable[..., Any]:
        """Build the combined module and semantic dispatcher, lower it,
        JIT-compile, and return a callable executor.

        Called initially from ``__init__`` (with zero sub-modules) and
        after every ``compile()`` to pick up newly added specializations.
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
                return result
            count = self._output_spec.num_leaves
            leaves = [] if count == 0 else [result] if count == 1 else list(result)
            return pytree.tree_unflatten(leaves, self._output_spec)

        signature = inspect.signature(self.fn)
        ffi_signature = signature.replace(
            parameters=[
                parameter.replace(default=_convert_input(parameter.default))
                for parameter in signature.parameters.values()
            ]
        )
        return tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
            invoke, ffi_signature
        )

    def _build_dispatcher(self, combined: ir.Module) -> None:
        """Try specializations in order, retrying only successful Exception payloads."""
        with self.ctx, combined.operation.location, ir.InsertionPoint(combined.body):
            any_type = tvm_ffi_d.AnyType.get()
            bool_type = tvm_ffi_d.BoolType.get()
            parameter_types = [tvm_ffi_d.PackArgumentsType.get()]
            dispatcher = tvm_ffi_d.PackFuncOp(
                self.fn.__name__,
                ir.TypeAttr.get(ir.FunctionType.get(parameter_types, [any_type])),
            )
            entry = dispatcher.add_entry_block()

            def attempt(index: int) -> tuple[ir.Value, ir.Value]:
                if index == len(self._sub_modules):
                    rejection = tvm_ffi_d.exception("GuardMatch")
                    return (
                        tvm_ffi_d.constant_bool(True),
                        tvm_ffi_d.upcast(any_type, rejection),
                    )
                [arguments] = entry.arguments
                success, payload = tvm_ffi_d.pack_call(
                    ir.FlatSymbolRefAttr.get(f"{self.fn.__name__}_{index}"),
                    arguments,
                    results=[bool_type, any_type],
                )
                rejected = tvm_ffi_d.to(
                    tvm_ffi_d.isa(
                        payload, ir.TypeAttr.get(tvm_ffi_d.ExceptionType.get())
                    )
                )
                retry = BINARY_ARITHMETIC[operator.and_](success, rejected)
                outcome = scf.IfOp(native(retry), [bool_type, any_type], has_else=True)
                with ir.InsertionPoint(outcome.then_block):
                    scf.yield_(attempt(index + 1))
                with ir.InsertionPoint(outcome.else_block):
                    scf.yield_([success, payload])
                return tuple(outcome.results)

            with ir.InsertionPoint(entry):
                success, payload = attempt(0)
                tvm_ffi_d.return_(success, payload)
