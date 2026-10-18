<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Operator adaptation

FX is imported directly to semantic TVM FFI and native MLIR operations. Start
with the captured FX graph and the dispatch schema, then choose the layer that
owns the behavior. Read [Architecture.md](Architecture.md) before changing the
import or lowering pipeline.

## ATen operators

`ffi/lib/aten/atengen.py` enumerates dispatcher schemas and generates registered
`trident.aten.<name>[.<overload>]` wrappers. Do not edit generated wrappers.
`TridentFxImporter._call_function` dispatches to `_build_aten`, which selects
the applicable ATen overload, resolves schema arguments, fills defaults and
converts layout and memory-format enums. `build_arguments` automatically converts
Python `torch.dtype` arguments to ATen's integer enum in the same ordered
argument traversal, including nested list/tuple elements. Existing SSA dtype
values and ordinary dtype constants retain their TVM FFI representation.
The same current-node handler constructs metadata operations or emits a checked
`tvm_ffi.FunctionCall`, returning the node's single Result, tuple of Results, or
unit Result. Most ATen operators therefore need no individual import
pattern. Tensor aliases and mutations follow the boxed ATen call directly; the
frontend does not functionalize the graph.

Check membership in `torch._C._dispatch_get_all_op_names()` before emitting a
registry call. A JIT schema alone does not establish dispatch support. Scalar
shape arithmetic belongs in native SSA, including comparisons, floor division,
remainder and boolean operations. Metadata reads use `tvm_ffi.tensor.size`,
`stride`, `dim`, `device`, `dtype` or `storage_offset`. Dtype values must be
converted to ATen's integer enum when the schema expects it.

Layout and memory-format constants are converted at import time through
`trident.runtime.torch_enum_to_int`. This TVM FFI helper reads the native enum
from PyTorch's internal Python types, avoiding a frontend numeric mapping.
It must be built against the installed PyTorch; these interfaces are not part
of PyTorch's stable ABI. The generated IR receives an ordinary integer constant.

For a new operator:

1. Inspect the raw exported graph and schema, including defaults and overloads.
2. Verify the generated wrapper exists, or add an import/lowering rule for an
   operation that has no dispatch schema.
3. Add public execution checks under `test/e2e`, organized by operator.
   `test/unit/test_guards.py` is reserved for guard parsing and expression IR.
   When dialect or lowering behavior changes, add a focused MLIR test at that
   layer. Existing generic dispatch coverage can be reused for an
   operator that needs no new lowering; avoid repeating its packed ABI checks.
4. Run the focused tests, full Python suite and core Lit target.

When adapting a mutation, verify both input writeback and output storage alias
on warmup and cached calls. Check specialization reuse so the repeated call
cannot silently exercise warmup again. Use owned input tensors when testing
cache reuse: `_base` guards conservatively force a miss. Non-contiguous inputs
can retain their layout with `clone(memory_format=torch.preserve_format)`.
When adapting a factory or clone, verify dtype/device and allocation behavior.
Do not compare uninitialized factory contents. Multi-result calls return an
FFI Array; retrieve each result with its semantic type. Python output container
structure is restored from its pytree specification.

## Semantic IR and ABI

FFI operations accept semantic FFI operands. Use `tvm_ffi.to` for native scalar
boxing, `tvm_ffi.get` for scalar extraction, and `tvm_ffi.upcast` for widening an ABI
value to Any or a compatible union. Use `tvm_ffi.downcast` to narrow Any/Union;
its result type determines the runtime type-tag assertion. A mismatch calls `abort()`.
The casts exclude native Function/Object handles and unions containing them.
Do not introduce frontend types into these
operations. Every checked registry call must check lookup status, a non-null
handle and call status before using its result.

Stateless scalar arithmetic, boxing and leaf constants live in
`python/trident/ir.py` and return ordinary values, as do schema and metadata
helpers. Independent fallible construction helpers return complete Results.
Callers use explicit `with bind(..., result_types=...) as operation` and access
successful payloads through `operation.arguments`, one per input in input order,
including None payloads. Only complete `operation.outputs` Results may leave a
successful bind body. Do not return bare payloads from these scopes. MLIR verification
checks `monad.yield` termination and matching Result types.

`build_call` checks global lookup status, a non-null handle and call status,
returning a complete Result without consuming a callback. `build_arguments`
constructs nested list/tuple arguments in order and returns Results, leaving
operator execution and subsequent graph traversal to its caller. Keep pure
argument prefixes iterative. Wrap successful values with the shared `pure`
builder. It infers types by default; with declared Result types it boxes native
scalars and widens Any/Union payloads at the boundary. The shared
`trident.ir.bind` function remains the only construction context manager and
restores insertion points on exceptions.

`visit(nodes, graph_module, outputs, result_types=result_types)` consumes a
shared FX iterator. Pure nodes stay in a loop. Node handlers return ordinary
structures containing Result leaves; the visitor binds
them, restores that structure and directly recurses into `visit` for subsequent
nodes. None Result payloads become Python `None`, consuming their corresponding
block arguments. Traversal keeps its working mappings, as does `import_program`. Use a fresh importer for
each program. Preserve ATen argument conversion,
result extraction and kernel clone/grid/launch order. Cond branches use
independent child importers sharing the module. Input extraction caches
complete, scope-checked Results retained until the table is discarded or
explicitly cleared. Guards and backend input flattening use fresh tables;
construction helpers do not automatically clear caches on return or exception.
Discard failed importers and their input tables; exceptional reuse is not
guaranteed. Preserve insertion-point restoration. Guards and the backend
explicitly bind their construction Results rather than passing continuation
callbacks.

FFI functions declare Result return types, and `tvm_ffi.return` matches those
types exactly. Every Result requires one payload type. Use `monad.pure` for
successful values and `Result<!tvm_ffi.none>` with `tvm_ffi.constant.none` for
no-data returns. Empty argument builders retain a None Result success marker;
pass only the original number of arguments to calls, including zero for empty
lists. Actual None arguments remain ordinary arguments. Use typed `pure`
construction for Any/Union widening. Runtime errors propagate through return
and packed ABI status.

Use the concrete Python type bindings when constructing or inspecting MLIR
types: for example, `tvm_ffi_d.TensorType.get()` and
`isinstance(value.type, tvm_ffi_d.TensorType)`. For native integers, inspect both
`IntegerType.width` and `IntegerType.is_signless` when the ABI requires a specific
signless width. Build unions with `UnionType.get(types, context=...)`; invalid,
duplicate or cross-context members raise `ValueError`. Type printing is for IR
text and diagnostics, not type dispatch.

The lowering pipeline decomposes binds into `monad.unpack` and `scf.if`, then
runs `lower-tvm-ffi-monad` after helper inlining and before ownership analysis.
Only successful branches restore widened payloads with `tvm_ffi.downcast`.

Ownership is handled by operation interfaces and the deallocation pass before
ABI lowering. Mark an allocation, borrow, alias or transfer accurately rather
than inserting reference-count calls in the importer. Extend semantic operations
and their lowering under `core/include/trident/core/Dialect/TVMFFI/` and
`core/lib/Conversion/` when an existing operation cannot represent the behavior.
Prefer DRR, then PDLL, for ordinary rewrites; use a C++ conversion pattern for
legality, type conversion or materialization requirements.

When debugging native calls, inspect LLVM's cached `TVMFFIFunctionGetGlobal`
names and compare them with the runtime registry. Calling a non-dispatch scalar
schema or supplying a dtype object to an integer schema can fail at the ABI
boundary even if import and IR verification succeeded.

`tvm_ffi.monad.unpack` exposes one Result as a success `i1` followed by its
payload. Payload types compatible with `tvm_ffi.upcast` to Any produce
`!tvm_ffi.any`: success preserves the value and failure produces None.
Other payload types, including native scalars and Function/Object handles,
remain unchanged and may only be used when success is true. Unpack neither
clears runtime errors nor transfers ownership. Result payload types remain
unrestricted except for the existing prohibition on nested Results.

## Symbolic shapes

Bind tensor dimensions and exported scalar inputs to runtime SSA values.
Evaluate expressions from these bindings; do not turn a SymInt hint into a
runtime constant. Add tests that reuse one compiled specialization with different
shapes and check both IR data flow and numerical results. Unsupported expressions
must report an import error. Triton's binder may use specialization hints to
select a kernel, but launch dimensions and runtime operands must use SSA values.

## Triton and kernel launch adaptation

`python/trident/importer.py` reuses the compiled Triton cache after warmup,
imports `gpu.binary`, and emits `tvm_ffi.kernel_launch`.
`TridentFxImporter._build_kernel` performs cache and autotuner lookup and
builds parameter attributes. Its recursive `_build_launch` constructs
clones in order, inserts the module-level binary and emits the launch. The
handler returns a dictionary of complete Results for cloned output tensors, or
a unit Result for a launch without outputs; the graph visitor binds these leaves
before importing subsequent nodes. Each source parameter has a constant or variable
specialization attribute. The latter records native ABI type and optional
divisibility; constexpr operands are checked and omitted from the native kernel
call. Functional wrapper outputs clone only the specified tensors before launch.

`ConvertTVMFFIToGPU` runs after inlining, Result expansion and SCF-to-CFG,
and before ownership deallocation.
It extracts native tensor data pointers and scalars from semantic operands,
validates specializations, obtains the current stream through TVM FFI, and emits
`gpu.launch_func`. A specialization failure must return the existing GuardMatch
exception through the enclosing wrapper's Any/union result. In a monadic wrapper,
wrap that exception in a successful Result at the return boundary rather than
reporting a runtime FFI failure.

Add a kernel launch or specialization Lit test and a cached Python execution
test under `test/e2e` for changes to this path. Use hook-free `triton.Config`
definitions in JIT examples because Dynamo export cannot capture arbitrary
Triton callbacks.
