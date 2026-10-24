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
shape arithmetic uses the shared semantic builders, including comparisons,
floor division, remainder and boolean operations. Python builders in
`python/trident/ir.py` own type unification, native extraction, conversion,
computation and result boxing. Extend their shared operation mappings rather
than adding C++ scalar operations or lowering patterns. Temporary native SSA
must be wrapped back into TVM FFI values before returning to the frontend.
Bool-to-Float conversion must be unsigned. The Python `logical_not` builder
compares against zero using ordered floating equality, so NaN remains truthy
and both floating zeros are falsy.
Metadata reads use `tvm_ffi.tensor.size`,
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
3. Add public execution checks to the matching capability module under
   `test/e2e` (operators, factories, storage or dynamic behavior).
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
helpers. The Python `power` builder handles nonnegative constant exponents;
source-level reductions compose the same scalar builders. Independent fallible
construction helpers return complete Results.
Callers consuming Results use `with BindScope(result_types, inputs) as scope`
and `scope.arguments` to expose one successful payload per input in input order,
including None payloads. Finish construction with `scope.yield_(results)`, which
terminates only this bind and returns nothing. Read `scope.results` for complete
Results after construction. Only these complete Results may leave the successful
scope; do not return bare payloads. MLIR verification checks `monad.yield`
termination and matching Result types.

`trident.ir.checked_call` checks global lookup status, a non-null handle and call status,
returning a complete Result without consuming a callback. `build_arguments`
constructs nested list/tuple arguments in order and returns Results, leaving
operator execution and subsequent graph traversal to its caller. Keep pure
argument prefixes iterative. `pure` borrows semantic payloads and constructs independent
Results, including repeated payload occurrences. It infers CABI types and
widens Any/Union payloads at declared boundaries. `make_result_types` accepts
CABI types directly; native scalar payloads and container elements are rejected.
Use `constant` for all scalar constants and the shared arithmetic and metadata
builders for frontend computations. Every ordinary builder returns TVM FFI
values, and symbol/value tables store those values without native extraction.
`native`/`tvm_ffi.get` is limited to temporary local arithmetic, metadata
adaptation and native SCF/CF/GPU interfaces; local arithmetic results are
immediately wrapped with `tvm_ffi.to`. Bind arguments remain CABI values.
`BindScope` borrows its inputs; scope contexts restore insertion points on normal and exceptional exits.

Use `ScopeStack` for FX traversal and other loops that accumulate a dynamic
chain of binds: `stack.bind(result_types, inputs)` exposes payloads, and
`stack.yield_(values)` drains scopes in reverse order, forwards complete Results
and restores insertion points. An empty stack forwards values unchanged. Child
stacks leave parent scopes open. Use `BindScope` directly for a single bind.
Build simple SCF branches and function bodies with `ir.InsertionPoint` and
explicit `scf.yield_`, `func.return_` or `tvm_ffi.return_`; they do not need Scope
wrappers. Context exits do not generate terminators, including on construction
failure.
Decomposition automatically retains semantic exported references before releasing
local owned references. Neither resource operations nor exits need attributes.

`visit(nodes, graph_module, outputs, result_types=result_types)` consumes an
FX iterator in a loop. Node handlers return ordinary structures containing Result
leaves; the visitor binds them through local `BindScope` contexts, restores the
structure and continues importing subsequent nodes in the successful scope. None Result payloads
become Python `None`, consuming their corresponding block arguments. Traversal
keeps its working mappings, as does `import_program`. Use a fresh importer for
each program. Preserve ATen argument conversion,
result extraction and kernel clone/grid/launch order. Cond branches use
independent child importers sharing the module. `InputTable.extract` and
`extract_many` own path extraction and ordered input reconstruction; `trident.ir.container_get` owns typed container extraction.
Guards use these shared builders directly and do not create or receive a graph
importer. Input extraction constructs fresh complete Results at each call site,
without caching. Explicit root and consumer-local payload bindings must remain
lexically visible; MLIR verification checks SSA visibility. Automatic semantic
management supplies lifetimes, and returned resources get independent references.
Guards and backend input flattening use fresh tables; construction helpers do not
automatically clear importer mappings on return or exception.
Discard failed importers and their input tables; exceptional reuse is not
guaranteed. Preserve insertion-point restoration. Guards and the backend
explicitly bind their construction Results rather than passing continuation
callbacks. Bind accepts independent input and output counts, including zero.
An empty input list executes the body unconditionally; an empty output list is
for effects only and ends with an empty `monad.yield`. The Python `BindScope` builder
supports the same counts; `pure([])` still constructs the existing None Result.

FFI functions declare exactly one C ABI payload type. `tvm_ffi.return` takes
a boxed success boolean and an Any payload. Pack multiple values into a semantic
tuple payload; generic no-data functions can declare `!tvm_ffi.none`. Generated
graph functions declare Any to admit both ordinary payloads and GuardMatch.
Graph computation and construction helpers use complete Results internally;
graph functions pack outputs at their TVM FFI boundary. Every Result requires
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
and expands local packed calls directly into LLVM control flow. Success returns
separate `(true, payload)` results; failure returns `(false, None)` without
reading the output slot or clearing the runtime error. The Any result owns the
returned payload reference; semantic calls receive automatic Any cleanup, while
Bool needs none. Local call lowering constructs no Array and performs no
registry lookup or temporary reference adjustment. Monad lowering and
`LowerToPack`/`TVMFFIToFunc` contain no packed call expansion, and LLVM conversion
receives no Result types.

Run `lower-to-pack` after decomposition and cleanup, then
`convert-tvm-ffi-to-func`. The first pass replaces all typed function bodies with
same-name packed entries, including functions without `emit_tvm_ffi_abi`.
Parameters must be decomposed indexed Base types or CAny. Count and tag guards
precede the original CFG; rejection returns successful GuardMatch. Successful
original returns check their declared payload type before widening to CAny.
The second pass lowers only packed entries to `__tvm_ffi_<name>` native ABI
functions; it creates no separate typed implementation or wrapper call.

Ownership contracts are implemented by semantic lowering. Decomposition selects
producer ownership through explicit helper calls and template arguments. Conversion legality
rejects unsupported semantic region boundaries. Synchronous captures borrow outer
references without a parent-operation whitelist; local producers register cleanup
inside their region. Defer lowering validates cleanup region structure.
Asynchronous reference lifetimes are outside the automatic ownership contract.
Base-only and native regions remain manually managed.
All Opaque types become Base, so type legality alone drives conversion; no
replacement-operation ownership tracking is needed. Creation, semantic calls and container
extraction acquire independent object references; casts, native extraction,
function parameters, internal CFG parameters and captures borrow. Forwarded
region entry arguments own one received reference per Opaque slot. Call result
ownership follows the original result type regardless of the function handle type. Base results remain manually managed,
including calls generated by container decomposition.
Custom decomposition patterns derive directly from
`mlir::OpConversionPattern<SourceOp>`. Borrowing patterns use the driver-provided
rewriter's replacement methods; producers that acquire independent references
register cleanup and replace their operations directly through that rewriter.
Representation-only patterns use `ConvertGenericOp<Op, OwnsResults>`;
`OwnsResults` defaults to false and producer registrations explicitly pass true.
The replacement path is selected at compile time. Producer patterns register
local Dec operations for original Opaque results in result order in a
defer region, so registration participates in conversion rollback. Value
replacements register cleanup at the current insertion point; operation
replacements place it after the replacement operation. All producer patterns preserve
ownership registration without a runtime operation-kind check.
Function returns emit an Inc for every original Opaque
operand in operand order. Dedicated if, for and while conversion patterns
insert retains directly at their operation-specific entry and exit positions.
Loop entry arguments and operation results register corresponding defer releases. This includes the
zero-trip parent-to-result path of `scf.for`, and both outcomes of
`scf.condition` without retaining the predicate. Use original
operation types with remapped values; do not trace provenance or infer ownership
from converted Base types or generated runtime calls. Original Base results,
including Base-typed container extractions, stay manually managed. Exporting a
manually managed value as Opaque still retains an independent boundary reference.
Run `lower-tvm-ffi-defer` immediately afterward: it clones deferred bodies at
dominance-scope exits, preserving body order. All export retains precede local
releases. Relative order between defers at the same exit is unspecified; operations
requiring an order must share one body.
Repeated exports acquire repeated references.
Directly exported producer results also retain each outgoing slot and receive
local cleanup. There is no implicit producer-reference transfer optimization.

Parent SCF patterns read original semantic types before converting region
signatures with the conversion rewriter. They handle initial-value retains,
entry argument cleanup and result cleanup, leaving child operations to their
own conversion patterns. Independent yield and condition patterns classify
ownership from original operand types and retain remapped adaptor payloads.
Yield ownership is limited to if, for and while boundaries; condition excludes
its predicate. There is no generic region analysis, successor-demand comparison
or shared ownership plan. Use the concrete operation's builder, preserve its
attributes and properties, and move existing regions through the conversion
rewriter. Do not directly mutate block argument or operation result types.
New region operations carrying semantic boundary values need their own
ownership conversion patterns. Patterns retain original Opaque slots directly; shared helpers
register deferred releases in slot order.
Runtime containers retain their elements and release them with the container.
Both low-level `MonadPureOp` and Python `pure` borrow payloads consistently.

Run monad lowering before decomposition, then `lower-tvm-ffi-defer`, all before
inlining or SCF-to-CFG conversion. Run decomposition only once. Automatically
registered cleanup supports multi-block function CFGs, `scf.for`, `scf.if` and
`scf.while`.
A registration remains active in blocks dominated by its registering block.
Cleanup runs on edges leaving that scope, on edges back to the registering block
before the next iteration, and at return/yield exits. Relative order between
defers at the same exit is unspecified. Edge blocks preserve branch
operands and path-specific execution. CFG block arguments remain borrowed:
forwarding a value neither transfers ownership nor extends its owner's lifetime.
Uses after the owner's dominance scope ends require an independently held
reference. Region entry arguments are distinct from these internal CFG
parameters: they receive references from the region transfer protocol and release
them on each region execution, after outgoing references have been preserved.
Direct Base operations and their explicit Inc/Dec remain manually managed, including
lowering-local temporary handles. Do not mix manual reference consumption with
automatically managed semantic values.

Explicit defer lowering supports any SSACFG region with block terminators,
without a parent-operation or exit whitelist. MLIR dominance analysis must
establish SSA semantics; graph regions and unknown single-block regions are
rejected. CFG branches implement `BranchOpInterface`; terminators without CFG
successors run all active cleanups before ending the region execution, including
`scf.condition` on either outcome. Each execution has independent registrations.
Forwarded values remain borrowed and require independently held references after
cleanup. This does not extend decomposition's automatic ownership. Unsupported
region structure and CFG branches within active scopes are diagnosed before
changing cleanup or CFG edges. Defer bodies still require one block without
arguments or nested regions. Unreachable registrations are removed without
cleanup copies. Empty bodies and edges requiring no cleanup may produce
forwarding blocks, which later canonicalization can remove. Automatic cleanup
registration may emit a single argument-free block containing only
`tvm_ffi.defer_yield` when no values need releasing; this is a valid empty defer.

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
and CStr alternatives after container normalization is deferred. Keep scalar elements in their TVM FFI types before
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
delegates parameter contracts to `_kernel_specialization`. Its iterative
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
`gpu.launch_func`. At the graph function exit, map graph Result failure to a successful
GuardMatch payload for the dispatcher. Private graphs use `tvm_ffi.func`;
do not add Union payloads or generate function exits in GPU lowering. Graph failure
currently means rejection because ordinary FFI calls assert on errors. Preserve
generic ABI runtime-error propagation outside this boundary.

Add a kernel launch or specialization Lit test and a cached Python execution
test under `test/e2e` for changes to this path. Use hook-free `triton.Config`
definitions in JIT examples because Dynamo export cannot capture arbitrary
Triton callbacks.

## Backend function construction

Construct the dispatcher with `tvm_ffi.pack.func`, and the guard wrapper and private graph with `tvm_ffi.func`
before running the common lowering pipeline. Use local packed calls between
layers. The dispatcher accepts one borrowed `!tvm_ffi.pack.arguments` view and retries only successful
Exception payloads; failed calls propagate their status without retrying.
Use `tvm_ffi.isa` for a non-trapping concrete runtime-tag test; `downcast`
remains an asserting cast. Packed argument counts and tags are checked by the converted callee entry before
reading slots or entering the implementation. Graph exits pack multiple successful outputs into a semantic
Tuple and translate internal specialization rejection to GuardMatch. Keep the
existing single-output and None conventions for Python pytree reconstruction.
