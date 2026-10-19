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
Callers consuming owned Results use explicit
`with bind(..., result_types=...) as operation` and access
successful payloads through `operation.arguments`, one per input in input order,
including None payloads. Only complete `operation.outputs` Results may leave a
successful bind body. Do not return bare payloads from these scopes. MLIR verification
checks `monad.yield` termination and matching Result types.

`trident.ir.checked_call` checks global lookup status, a non-null handle and call status,
returning a complete Result without consuming a callback. `build_arguments`
constructs nested list/tuple arguments in order and returns Results, leaving
operator execution and subsequent graph traversal to its caller. Keep pure
argument prefixes iterative. `pure` borrows payloads and requests an independent
reference for each Result through `tvm_ffi.retain_payloads`, then marks it for
cleanup. It infers types by default;
with declared Result types it boxes native scalars and widens Any/Union payloads at the boundary. `bind` borrows
its inputs; its builder unconditionally marks the bind operation for cleanup. For exits marked with `tvm_ffi.retain`, the cleanup pass retains
region yields and function returns to establish independent references for the
receiving scope before releasing local references. Result retains read
payloads only on success; scalars need no references. The bind builder restores
insertion points on exceptions.

Frontend exits construct `func.ReturnOp`, `tvm_ffi_d.ReturnOp`,
`scf.YieldOp` or `tvm_ffi_d.MonadYieldOp`, then set the attribute directly:

```python
from trident.ownership import CLEANUP_ATTR, RETAIN_ATTR

terminator = tvm_ffi_d.MonadYieldOp(results)
terminator.attributes[RETAIN_ATTR] = ir.UnitAttr.get()
```

The cleanup pass consumes the UnitAttr, retaining each reference-bearing operand
occurrence before releasing local references. Unmarked exits request no
automatic retains. Do not manually retain outputs for marked exits. Payload
retains requested by `pure` remain independent of export retains.

`visit(nodes, graph_module, outputs, result_types=result_types)` consumes a
shared FX iterator. Pure nodes stay in a loop. Node handlers return ordinary
structures containing Result leaves; the visitor binds
them, restores that structure and directly recurses into `visit` for subsequent
nodes. None Result payloads become Python `None`, consuming their corresponding
block arguments. Traversal keeps its working mappings, as does `import_program`. Use a fresh importer for
each program. Preserve ATen argument conversion,
result extraction and kernel clone/grid/launch order. Cond branches use
independent child importers sharing the module. `InputTable.extract` and
`extract_many` own path extraction and ordered input reconstruction; `trident.ir.container_get` owns typed container extraction.
Guards use these shared builders directly and do not create or receive a graph
importer. Input extraction caches borrowed complete Results; reuse requires a
cleanup-marked reference as well as lexical visibility. Returned resources get independent references. Guards and backend
input flattening use fresh tables;
construction helpers do not automatically clear caches on return or exception.
Discard failed importers and their input tables; exceptional reuse is not
guaranteed. Preserve insertion-point restoration. Guards and the backend
explicitly bind their construction Results rather than passing continuation
callbacks. Bind accepts independent input and output counts, including zero.
An empty input list executes the body unconditionally; an empty output list is
for effects only and ends with an empty `monad.yield`. The Python `bind` helper
supports the same counts; `pure([])` still constructs the existing None Result.

FFI functions declare exactly one Result return type, and `tvm_ffi.return`
matches that type exactly. Pack multiple values into a semantic tuple payload. Every Result requires one payload type. Use `monad.pure` for
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

The lowering pipeline decomposes binds into `monad.unpack` and `scf.if`,
retaining semantic Results until LLVM conversion. LLVM conversion lowers each
Result to one struct containing its converted payload and an i1 success bit,
and lowers `monad.pure`, `monad.fail` and `monad.unpack`. ABI return conversion
uses unpack and restores widened payloads only in successful branches with
`tvm_ffi.downcast`.

For operations producing independent owned references, construct the Op and
set `operation.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()` directly.
The `tvm_ffi.cleanup` UnitAttr schedules one release per reference-bearing result; scalar results are ignored. There are no
per-result count overrides. Semantic container construction and extraction
acquire cleanup markers when decomposed; other operations have no default
cleanup credits. Marking twice does not
add another release. Borrowed casts, unpacked payloads, block arguments and
metadata results must not acquire cleanup responsibility. Shared tensor storage
does not imply a borrowed reference. Ownership interfaces describe operation
contracts independently of explicit cleanup policy.

Python does not classify reference-bearing types. Direct cleanup attribute
assignment marks the operation, including scalar-only and zero-result
operations. The C++ cleanup pass privately classifies types after semantic Array/Tuple decomposition to
select results requiring releases, consuming scalar-only and zero-result markers
without reference operations. Input caches require lexical visibility and an
OpResult whose defining operation carries a cleanup UnitAttr before reuse,
regardless of result type.

`pure` requests a retain for each resource payload occurrence, including repeated values;
callers do not retain payloads separately before wrapping them. The low-level
`MonadPureOp` transfers references and requires explicit ownership management.
Construct region yields and function returns, then directly assign their
`tvm_ffi.retain` UnitAttr; the cleanup pass consumes the markers and
retains their reference-bearing operands. `pure` defers payload
retains until cleanup after container erasure. Result retains and releases use
`monad.unpack` and one `scf.if` success branch per Result so that each successful
payload is adjusted independently, even when other Results fail. Reference
operations accept widened Any payloads directly; native handles keep their types.

`decompose-tvm-ffi` erases containers and unconditionally marks container
runtime calls and converted FunctionCall operations for cleanup before `materialize-tvm-ffi-cleanup` runs. Cleanup executes
before inlining and SCF-to-CFG, directly emitting unpack and success branches
for Result reference adjustments. No second decomposition is needed. Cleanup
assumes well-formed input attributes and processes block exits only when their
immediate parent region has one block. Separate
payload-retain, export-retain and cleanup pattern stages run without folding or
dead-code elimination. Export retention handles
marked reference-bearing operands of `func.return`, `tvm_ffi.return`, `scf.yield` and
`tvm_ffi.monad.yield` before releasing local references in reverse order. There is no scope operation whitelist. Empty and
multi-block regions have no direct block-exit cleanup or export rewrites. Nested single-block
regions are processed independently when visited by the walk driver, which skips
unreachable blocks. Results of an operation owning a skipped region are still
processed in its selected parent block. Scalar-only and empty exits need no
retains. Only exits carrying `tvm_ffi.retain` request export retains.
Consumed export and cleanup markers are removed for idempotence, including
export markers on scalar-only and empty exits. Lowerings preserve cleanup
markers on replacement resource Ops and pending export markers on converted
exits. Generated failure exits need no export marker. Temporary references with explicit
releases remain unmarked. Constant specialization helpers follow the same rule
for their temporary containers.
Ownership interfaces remain operation contracts; cleanup does not infer aliases,
liveness or ownership across CFG edges.

Extend semantic operations
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

## Semantic containers

Use `ArrayType.get(element_type)` for lists and `TupleType.get(types)` for tuples.
Array and Tuple element types must inherit `TVMFFICABIType`; native Object
handles are outside CABI and cannot be container elements. Semantic Array and Tuple
inherit `TVMFFICOpaqueABIType` (and therefore `TVMFFIABIType`), preserving nested
containers. Decomposition erases both to the concrete Base `CArrayType`.
Array's C++ element parameter and accessor use `TVMFFICABIType`; parsing and
checked C/Python construction reject non-CABI elements. Native scalars and Results are not
valid container elements. Box native scalars before construction; heterogeneous and
empty lists use `ArrayType.get(AnyType.get())`. The ABI Array binding is
`CArrayType.get()`.

Emit `array.create`, `array.get` and `array.size` for lists; these return the
array, element and boxed integer types directly. Array get accepts a native
i64 index.
For tuples, `tuple.construct` directly returns a semantic Tuple and
`tuple.unconstruct` directly returns every element in order, without Results.
Use unconstruct results for static indexing and type information for static
length. Input guards must still check runtime arity through checked
`ffi.ArraySize` calls. Container decomposition asserts lookup status, a non-null
handle and call status, terminating on failure.
`decompose-tvm-ffi` uses one dialect conversion to expand checked runtime calls
and monadic binds while erasing container types to `c.array` throughout
signatures and regions.
Keep this pass before ABI and LLVM conversion. Direct runtime
ABI helpers may continue using `c.array`; frontend code uses semantic types.

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
delegates parameter contracts to `_kernel_specialization`. Its recursive
`_build_launch` constructs clones in order, resolves grid operands and emits the
launch; `_import_kernel_binary` installs the module-level binary once. The
handler returns a dictionary of complete Results for cloned output tensors, or
a unit Result for a launch without outputs; the graph visitor binds these leaves
before importing subsequent nodes. Each source parameter has a constant or variable
specialization attribute. The latter records native ABI type and optional
divisibility; constexpr operands are checked and omitted from the native kernel
call. Functional wrapper outputs clone only the specified tensors before launch.

`ConvertTVMFFIToGPU` runs after inlining and SCF-to-CFG, before ABI and LLVM
conversion. The launch returns `Result<None>`: matching inputs launch and return success;
rejection returns `monad.fail`, does not launch and does not exit the function.
Bind the launch Result explicitly so rejection suppresses later effects and
passes through normal cleanup.
It extracts native tensor data pointers and scalars from semantic operands,
validates specializations, obtains the current stream through TVM FFI, and emits
`gpu.launch_func`. At the Python-built graph-call boundary, map graph Result failure to a successful
GuardMatch payload for the dispatcher. Keep private graph signatures unchanged;
do not add Union payloads or generate function exits in GPU lowering. Graph failure
currently means rejection because ordinary FFI calls assert on errors. Preserve
generic ABI runtime-error propagation outside this boundary.

Add a kernel launch or specialization Lit test and a cached Python execution
test under `test/e2e` for changes to this path. Use hook-free `triton.Config`
definitions in JIT examples because Dynamo export cannot capture arbitrary
Triton callbacks.
