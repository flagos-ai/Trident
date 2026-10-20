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
    arith,
    func,
    llvm,
    scf,
    transform,
)
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.execution_engine import ExecutionEngine
from trident.ffi import Exception
from trident.ownership import CLEANUP_ATTR, RETAIN_ATTR

from .guards import parse_guards
from .importer import TridentFxImporter
from .input import InputTableBuilder
from .ir import bind, container_create, native, pure, tuple_type


class TridentGraphModule:
    """Imports a Torch FX graph into MLIR and executes through TVM FFI.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus an LLVM-level
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order - the first one that returns 0 (guard match) wins.
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

    def _alloca_tvmffi_any(
        self,
        index: int,
        payload: ir.Value,
    ) -> ir.Value:
        """Allocate a TVMFFIAny slot, fill it, and return the pointer.

        Initializes the slot with a complete ABI struct.
        """
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        type_ = llvm.StructType.get_literal([i32, i32, i64], context=self.ctx)
        slot = llvm.alloca(
            res=llvm.PointerType.get(),
            array_size=llvm.mlir_constant(ir.IntegerAttr.get(i64, 1)),
            elem_type=ir.TypeAttr.get(type_),
        )
        result = llvm.mlir_undef(type_)
        fields = (
            llvm.mlir_constant(ir.IntegerAttr.get(i32, index)),
            llvm.mlir_constant(ir.IntegerAttr.get(i32, 0)),
            payload,
        )
        for position, value in enumerate(fields):
            result = llvm.insertvalue(result, value, [position])
        llvm.store(result, slot)
        return slot

    @staticmethod
    def _build_graph_result(
        function: func.FuncOp,
        arguments: Sequence[ir.Value],
        rejection: ir.Value,
        result_type: ir.Type,
    ) -> list[ir.Value]:
        """Pack owned graph outputs or translate specialization rejection to GuardMatch."""
        graph_call = func.CallOp(function, arguments)
        graph_call.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
        graph_results = graph_call.results
        ready = arith.constant(ir.IntegerType.get_signless(1), True)
        for result in graph_results:
            unpack = tvm_ffi_d.MonadUnpackOp(result)
            ready = arith.andi(ready, unpack.success)
        outcome = scf.IfOp(ready, [result_type], has_else=True)
        outcome.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
        with ir.InsertionPoint(outcome.then_block):
            with bind(
                graph_results,
                result_types=[result_type],
            ) as invocation:
                if len(invocation.arguments) == 1:
                    results = pure(
                        invocation.arguments,
                        result_types=[result_type],
                    )
                else:
                    with bind(
                        container_create(
                            invocation.arguments,
                            tuple_type(
                                [value.type for value in invocation.arguments],
                            ),
                        ),
                        result_types=[result_type],
                    ) as packing:
                        yielded = pure(
                            packing.arguments,
                            result_types=[result_type],
                        )
                        terminator = tvm_ffi_d.MonadYieldOp(yielded)
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    results = [*packing.outputs]
                terminator = tvm_ffi_d.MonadYieldOp(results)
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
            terminator = scf.YieldOp(invocation.outputs)
            terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
        with ir.InsertionPoint(outcome.else_block):
            yielded = pure(
                [rejection],
                result_types=[result_type],
            )
            terminator = scf.YieldOp(yielded)
            terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
        return [*outcome.results]

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
        main_func: func.FuncOp = importer.import_program(
            exported_program, func_name=main_func_name
        )
        with self.ctx:
            main_func.attributes["sym_visibility"] = ir.StringAttr.get("private")
        module: ir.Module = importer.module

        torch._dynamo.reset()

        # Step 3: Wrap with tvm_ffi.func.
        flat_types: Sequence[ir.Type] = main_func.type.inputs

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
            ffi_func: tvm_ffi_d.FuncOp = tvm_ffi_d.func(
                tvm_ffi_name,
                ir.TypeAttr.get(ffi_type),
                emit_tvm_ffi_abi=ir.UnitAttr.get(self.ctx),
            )
            entry_block: ir.Block = ir.Block.create_at_start(ffi_func.body, param_types)

            with ir.InsertionPoint(entry_block):
                owned_op = tvm_ffi_d.ExceptionOp("GuardMatch")
                owned_op.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                exception = owned_op.result
                rejection = tvm_ffi_d.upcast(abi_result_type, exception)
                with bind(
                    parse_guards(gs).build(
                        input_builder, entry_block.arguments, self.ctx
                    ),
                    result_types=[ffi_result_type],
                ) as guards:
                    [predicate] = guards.arguments
                    branch = scf.IfOp(
                        native(predicate), [ffi_result_type], has_else=True
                    )
                    branch.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()
                    with ir.InsertionPoint(branch.then_block):
                        with bind(
                            input_builder.build(entry_block.arguments).extract_many(),
                            result_types=[ffi_result_type],
                        ) as inputs:
                            input_specs = exported_program.graph_signature.input_specs
                            input_values = inputs.arguments[: len(input_specs)]
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
                            yielded = self._build_graph_result(
                                main_func,
                                main_inputs,
                                rejection,
                                ffi_result_type,
                            )
                            terminator = tvm_ffi_d.MonadYieldOp(yielded)
                            terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                        terminator = scf.YieldOp(inputs.outputs)
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    with ir.InsertionPoint(branch.else_block):
                        yielded = pure([rejection], result_types=[ffi_result_type])
                        terminator = scf.YieldOp(yielded)
                        terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                    terminator = tvm_ffi_d.MonadYieldOp(branch.results)
                    terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
                unpacked = tvm_ffi_d.MonadUnpackOp(guards.outputs[0])
                success = tvm_ffi_d.to(unpacked.success)
                terminator = tvm_ffi_d.ReturnOp(success, unpacked.value)
                terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()

        self._sub_modules.append(module)
        self.executor = self.stub_compile()
        gc.collect()
        return warmup_result

    def stub_compile(self) -> Callable[..., Any]:
        """Build the combined module, lower it, add the LLVM dispatcher,
        JIT-compile, and return a callable executor.

        Called initially from ``__init__`` (with zero sub-modules) and
        after every ``compile()`` to pick up newly added specializations.
        """
        # 1. Merge all sub-modules into one (each is cloned to avoid
        #    repeated-merging hangs).
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

        # 2. Lower semantic TVM FFI and GPU IR to LLVM.
        with self.ctx:
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)",
            ).run(combined.operation)

        # 3. Build the LLVM dispatcher that tries each sub-function.
        with self.ctx, combined.operation.location, ir.InsertionPoint(combined.body):
            i32 = ir.IntegerType.get_signless(32)
            i64 = ir.IntegerType.get_signless(64)
            pointer = llvm.PointerType.get()
            any_type = llvm.StructType.get_literal([i32, i32, i64])
            name_type = llvm.StructType.get_literal([pointer, i64])
            packed_type = ir.Type.parse(
                "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr)>"
            )
            symbols = ir.SymbolTable(combined.operation)
            declarations = {
                "TVMFFIFunctionCall": packed_type,
                "TVMFFIFunctionGetGlobal": ir.Type.parse(
                    "!llvm.func<i32 (!llvm.ptr, !llvm.ptr)>"
                ),
                "TVMFFIObjectDecRef": ir.Type.parse("!llvm.func<i32 (!llvm.ptr)>"),
            }
            for name, type_ in declarations.items():
                if name not in symbols:
                    llvm.func(name, ir.TypeAttr.get(type_))
            dispatcher = llvm.func(
                f"__tvm_ffi_{self.fn.__name__}", ir.TypeAttr.get(packed_type)
            )
            entry = ir.Block.create_at_start(
                dispatcher.body, [pointer, pointer, i32, pointer]
            )
            in_ptr, out_ptr, num_args, ret_ptr = entry.arguments
            error = ir.Block.create_after(entry)
            done = ir.Block.create_after(error)
            exit_ = ir.Block.create_after(done, i32)

            call: Callable[[str, Sequence[ir.Value]], ir.Value] = (
                lambda name, arguments: llvm.call(i32, arguments, [], [], callee=name)
            )
            field: Callable[[ir.Value, int, ir.Type], ir.Value] = (
                lambda slot, index, type_: llvm.extractvalue(
                    type_, llvm.load(any_type, slot), [index]
                )
            )

            def string(name: str, value: str) -> ir.Value:
                with ir.InsertionPoint(combined.body):
                    llvm.mlir_global(
                        ir.TypeAttr.get(
                            ir.Type.parse(f"!llvm.array<{len(value)} x i8>")
                        ),
                        name,
                        ir.Attribute.parse("#llvm.linkage<internal>"),
                        constant=True,
                        value=ir.StringAttr.get(value),
                    )
                return llvm.mlir_addressof(pointer, name)

            with ir.InsertionPoint(entry):
                zero = llvm.mlir_constant(ir.IntegerAttr.get(i32, 0))
                one = llvm.mlir_constant(ir.IntegerAttr.get(i32, 1))
                zero_i64 = llvm.mlir_constant(ir.IntegerAttr.get(i64, 0))
                one_i64 = llvm.mlir_constant(ir.IntegerAttr.get(i64, 1))
                null = llvm.mlir_zero(pointer)
                handles = {}
                for name in ("trident.ffi.Exception", "trident.ffi.GetExceptionIndex"):
                    name_ptr = string(f"__trident_const_{name}", name)
                    name_value = llvm.mlir_undef(name_type)
                    name_value = llvm.insertvalue(name_value, name_ptr, [0])
                    name_value = llvm.insertvalue(
                        name_value,
                        llvm.mlir_constant(ir.IntegerAttr.get(i64, len(name))),
                        [1],
                    )
                    name_slot = llvm.alloca(pointer, one_i64, name_type)
                    llvm.store(name_value, name_slot)
                    handle_slot = llvm.alloca(pointer, one_i64, pointer)
                    call("TVMFFIFunctionGetGlobal", [name_slot, handle_slot])
                    handles[name] = llvm.load(pointer, handle_slot)
                index_slot = self._alloca_tvmffi_any(0, zero_i64)
                call(
                    "TVMFFIFunctionCall",
                    [handles["trident.ffi.GetExceptionIndex"], null, zero, index_slot],
                )
                exception_index = llvm.trunc(
                    i32,
                    field(index_slot, 2, i64),
                    ir.Attribute.parse("#llvm.overflow<none>"),
                )
                result_slot = self._alloca_tvmffi_any(0, zero_i64)
                kind = string(
                    "__trident_const_GuardMatchException", "GuardMatchException\0"
                )
                arguments = self._alloca_tvmffi_any(8, llvm.ptrtoint(i64, kind))
                call(
                    "TVMFFIFunctionCall",
                    [handles["trident.ffi.Exception"], arguments, one, result_slot],
                )

            previous = entry
            for index in range(len(self._sub_modules)):
                attempt = ir.Block.create_after(previous)
                check = ir.Block.create_after(attempt)
                with ir.InsertionPoint(previous):
                    failed = llvm.icmp(
                        llvm.ICmpPredicate.eq,
                        field(result_slot, 0, i32),
                        exception_index,
                    )
                    llvm.cond_br(failed, [], [], attempt, done)
                with ir.InsertionPoint(attempt):
                    call(
                        "TVMFFIObjectDecRef",
                        [llvm.inttoptr(pointer, field(result_slot, 2, i64))],
                    )
                    status = call(
                        f"__tvm_ffi_{self.fn.__name__}_{index}",
                        [in_ptr, out_ptr, num_args, result_slot],
                    )
                    success = llvm.icmp(llvm.ICmpPredicate.eq, status, zero)
                    llvm.cond_br(success, [], [], check, error)
                previous = check
            with ir.InsertionPoint(previous):
                llvm.br([], done)
            with ir.InsertionPoint(error):
                llvm.br([llvm.mlir_constant(ir.IntegerAttr.get(i32, -1))], exit_)
            with ir.InsertionPoint(done):
                llvm.store(llvm.load(any_type, result_slot), ret_ptr)
                llvm.br([zero], exit_)
            with ir.InsertionPoint(exit_):
                for handle in handles.values():
                    call("TVMFFIObjectDecRef", [handle])
                [status] = exit_.arguments
                llvm.return_(arg=status)

        # 4. JIT-compile and wrap.
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
            result = normalize(
                fn(
                    *(
                        normalize(
                            value,
                            lambda element: (
                                tvm_ffi.device(f"{element}")
                                if isinstance(element, torch.device)
                                else tvm_ffi.convert(element)
                                if isinstance(element, torch.dtype)
                                else element
                            ),
                        )
                        for value in values
                    )
                ),
                lambda value: (
                    torch.from_dlpack(value)
                    if isinstance(value, tvm_ffi.Tensor)
                    else value
                ),
            )

            if isinstance(result, Exception):
                return result
            leaves = [result] if self._output_spec.num_leaves == 1 else list(result)
            return pytree.tree_unflatten(leaves, self._output_spec)

        def normalize(
            value: Any,
            convert: Callable[[Any], Any],
        ) -> Any:
            if isinstance(value, (tvm_ffi.Array, list)):
                return [normalize(element, convert) for element in value]
            elif isinstance(value, tuple):
                return tuple(normalize(element, convert) for element in value)
            elif isinstance(value, dict):
                return {
                    key: normalize(element, convert) for key, element in value.items()
                }
            else:
                return convert(value)

        signature = inspect.signature(self.fn)
        ffi_signature = signature.replace(
            parameters=[
                parameter.replace(
                    default=(
                        tvm_ffi.device(f"{parameter.default}")
                        if isinstance(parameter.default, torch.device)
                        else tvm_ffi.convert(parameter.default)
                    )
                )
                if isinstance(parameter.default, (torch.device, torch.dtype))
                else parameter
                for parameter in signature.parameters.values()
            ]
        )
        return tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
            invoke, ffi_signature
        )
