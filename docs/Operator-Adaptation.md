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

The Python frontend constructs object-free Base types and Opaque types only.
Use semantic Tensor, Exception, Function and Str; their object-bearing Base
counterparts are CTensor, CException, CFunction and CStr. Array/Tuple lower to
CArray, and Any/Union/Str lower to the dynamic tagged Base type CAny.
CAny has no single type index and remains manually managed in handwritten Base IR.
Keep C-prefixed types and native Object handles in C++ lowering
and low-level tests. Base bindings remain available for inspection. Add C/Python
construction, inspection and TypeID bindings whenever adding a dialect type.

Opaque/Base separation selects the underlying representation and manages
semantic reference lifetimes automatically. Keep frontend IR semantic and let
`decompose-tvm-ffi` materialize export retains and deferred local releases; do not add
ownership attributes. Handwritten Base IR remains manually managed.

FFI operations accept semantic FFI operands. Use `tvm_ffi.to` for native scalar
boxing, `tvm_ffi.get` for scalar extraction, and `tvm_ffi.upcast` for widening an ABI
value to Any or a compatible union. Both casts accept conversion to the same
type as a no-op.
Use `tvm_ffi.downcast` to narrow Any/Union. After decomposition, a concrete
indexed Base result type determines the runtime type-tag assertion; a mismatch
calls `abort()`. Union/Str targets and packed function parameters and results
currently become unconstrained CAny. Preserving their finite alternatives for
explicit downcast checks and boundary validation is deferred.
The casts exclude native Object handles. Semantic Function values retain the
existing CFunction pointer boxing and extraction behavior.
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
argument prefixes iterative. `pure` borrows payloads and constructs independent
Results, including repeated payload occurrences. It infers boxed CABI types,
boxes native scalars, and widens Any/Union payloads at declared boundaries.
`make_result_types` applies the same scalar type boxing. Bind arguments are CABI
values; use `native` or `tvm_ffi.get` for native computation. `bind` borrows its
inputs and restores insertion points on exceptions.

Build `func.return`, `tvm_ffi.return`, `scf.yield` and `monad.yield` normally.
Decomposition automatically retains semantic exported references before releasing
local owned references. Neither resource operations nor exits need attributes.

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
importer. Input extraction caches borrowed complete Results; reuse requires lexical
visibility, with lifetime supplied by automatic semantic management. Returned resources get independent references. Guards and backend
input flattening use fresh tables;
construction helpers do not automatically clear caches on return or exception.
Discard failed importers and their input tables; exceptional reuse is not
guaranteed. Preserve insertion-point restoration. Guards and the backend
explicitly bind their construction Results rather than passing continuation
callbacks. Bind accepts independent input and output counts, including zero.
An empty input list executes the body unconditionally; an empty output list is
for effects only and ends with an empty `monad.yield`. The Python `bind` helper
supports the same counts; `pure([])` still constructs the existing None Result.

FFI functions declare exactly one C ABI payload type. `tvm_ffi.return` takes
a boxed success boolean and an Any payload. Pack multiple values into a semantic
tuple payload; no-data functions declare `!tvm_ffi.none`. Private graph helpers
and construction helpers continue to use complete Results. Every Result requires
one CABI payload type; use `monad.pure` for success and explicit None payloads
for no-data Results. Empty argument builders retain a None Result success marker;
pass only the original number of arguments to calls, including zero for empty
lists. Actual None arguments retain their positions. Runtime errors propagate
through return and packed ABI status.

Use the concrete Python type bindings when constructing or inspecting MLIR
types: for example, `tvm_ffi_d.TensorType.get()` and
`isinstance(value.type, tvm_ffi_d.TensorType)`. For native integers, inspect both
`IntegerType.width` and `IntegerType.is_signless` when the ABI requires a specific
signless width. Build unions with `UnionType.get(types, context=...)`; invalid,
duplicate or cross-context members raise `ValueError`. Type printing is for IR
text and diagnostics, not type dispatch.

`lower-tvm-ffi-monad` converts Results to semantic `tuple<bool, any>` and
binds to tuple extraction and `scf.if`. `tvm_ffi.monad.call` returns a complete
`Result<T>` matching its local callee's return type. Its lowering emits
`tvm_ffi.call` with separate `!tvm_ffi.bool` and `!tvm_ffi.any` results, explicitly
constructs a tuple borrowing those results. Decomposition automatically releases
the call payload and local tuple, retaining any exported tuple first. Result attributes apply
to the call's payload, with an empty success-result entry. Local callee resolution, argument matching and the `emit_tvm_ffi_abi`
requirement are deferred to lowering for unlowered callees. Registry
calls continue to use `tvm_ffi.FunctionCall`.

`decompose-tvm-ffi` preserves the call's two results and converts container
operands and explicit tuples to CArrays. `TVMFFIToLLVM` requires this decomposition
and expands packed wrappers directly into LLVM control flow. Success returns
separate `(true, payload)` results; failure returns `(false, None)` without
reading the output slot or clearing the runtime error. The Any result owns the
returned payload reference; semantic calls receive automatic Any cleanup, while
Bool needs none. Local call lowering constructs no Array and performs no
registry lookup or temporary reference adjustment. Monad lowering and
`TVMFFIToFunc` contain no packed call expansion, and LLVM conversion receives no
Result types.

Ownership contracts are implemented by semantic lowering. Decomposition selects
producer ownership through explicit pattern inheritance; scope validation checks
region structure independently.
All Opaque types become Base, so type legality alone drives conversion; no
replacement-operation ownership tracking is needed. Creation, semantic calls and container
extraction acquire independent object references; casts, native extraction,
parameters and captures borrow. Call result ownership follows the original result
type regardless of the function handle type. Base results remain manually managed,
including calls generated by container decomposition.
Custom decomposition patterns use `ReplacementConversionPattern<SourceOp>`
for borrowing results and its `OwnershipConversionPattern<SourceOp>` subclass
for producers that acquire independent references. Representation-only patterns
select this base through their template parameter at registration. Use the
selected base's protected static `replaceOp` and `replaceOpWithNewOp` helpers
with the driver-provided rewriter. The ownership subclass automatically registers
local Dec operations for original Opaque results in a defer region, so
registration participates in conversion rollback. All three replacement helper
forms on that subclass preserve ownership registration without a runtime
operation-kind check. Each return/yield
pattern emits an Inc for every original Opaque operand in operand order,
independent of its source. Each Opaque
`scf.if` result owns the reference exported by the selected branch. Use original
operation types with remapped values; do not trace provenance or infer ownership
from converted Base types or generated runtime calls. Original Base results,
including Base-typed container extractions, stay manually managed. Exporting a
manually managed value as Opaque still retains an independent boundary reference.
Run `lower-tvm-ffi-defer` immediately afterward: it moves deferred bodies before
return/yield in reverse registration order, preserving body order. All export
retains precede local releases, which run in reverse definition and result order.
Repeated exports acquire repeated references.
Runtime containers retain their elements and release them with the container.
Both low-level `MonadPureOp` and Python `pure` borrow payloads consistently.

Run monad lowering before decomposition, then `lower-tvm-ffi-defer`, all before
inlining or SCF-to-CFG conversion. Run decomposition only once. Automatically
registered cleanup is block-local and supports captured values, but not nested
regions in its body.
Automatic management supports single-block function and `scf.if` regions.
Unsupported control-flow ownership produces a diagnostic;
there is no CFG ownership inference or early-release optimization. Direct Base
operations and their explicit Inc/Dec remain manually managed, including
lowering-local temporary handles. Do not mix manual reference consumption with
automatically managed semantic values.

Defer lowering uses the same single-block lexical cleanup scope: only
`tvm_ffi.func`, `func.func` and `scf.if` regions with `tvm_ffi.return`,
`func.return` or `scf.yield` exits are supported. Multi-block regions,
unsupported enclosing operations and unsupported exits are diagnosed before
moving cleanup. Inner branch cleanup runs before leaving the branch, while
outer cleanup stays before the outer block terminator.

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
Function payloads also widen to Any through CFunction pointer boxing. Unpack neither clears runtime errors nor
transfers ownership. Result payloads must be `TVMFFICABIType`; native scalars,
native Object handles and nested Results are rejected by parsing and checked
C/Python construction. The C++ payload parameter and accessor use
`TVMFFICABIType`. Function remains a valid CABI payload and supports Any casts through
CFunction pointer boxing and extraction.

## Semantic containers

Use `ArrayType.get(element_type)` for lists and `TupleType.get(types)` for tuples.
Array and Tuple element types must inherit `TVMFFICABIType`; native Object
handles are outside CABI and cannot be container elements. Semantic Array and Tuple
inherit `TVMFFICOpaqueABIType` (and therefore `TVMFFIABIType`), preserving nested
containers. Decomposition erases both to the concrete Base `CArrayType`.
Array's C++ element parameter and accessor use `TVMFFICABIType`; parsing and
checked C/Python construction reject non-CABI elements. Native scalars and Results are not
valid container elements. Frontend strings use Opaque Str, including constants, container elements and
Result payloads. Decomposition erases Str to CAny; checking the RawStr, SmallStr
and CStr alternatives after container normalization is deferred. Box native scalars before
construction; heterogeneous and
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
`decompose-tvm-ffi` preserves local calls and uses a dialect conversion to expand checked container runtime calls while erasing container types to `c.array` throughout
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

Kernel constant specializations compare ABI values directly, including semantic
strings; only variable launch operands require native materialization. Constant
operands participate in guards and are omitted from the GPU launch arguments.

`ConvertTVMFFIToGPU` runs before monad lowering, container decomposition and
cleanup. The launch returns `Result<None>`: matching inputs launch and return success;
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
