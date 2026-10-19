<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Architecture

Trident captures Python tensor programs with Dynamo and `torch.export`, imports
FX directly into MLIR, and executes the lowered program through TVM FFI. The
frontend owns the import; the build depends on LLVM/MLIR, PyTorch, CUDA and TVM
FFI. The LLVM revision remains pinned in the top-level CMake file.

## Frontend and symbolic values

`python/trident/backend.py` captures a graph and its Dynamo guards. It exports
without running functionalization or decompositions, executes a warmup to populate
Triton's compilation cache, and calls `TridentFxImporter.import_program`.
Warmup is the first execution, including its mutations; cached calls execute the
compiled program. A private `func.func` holds the graph, and a public
`tvm_ffi.func` checks guards and exposes the packed ABI. The Python executor
converts runtime tensors through DLPack and reconstructs the output pytree.
The LLVM dispatcher tries packed specializations in order, reads their ABI
result tags, releases guard exceptions before retrying, and propagates runtime
call failures. ABI field reads and calls share local builders.

`importer.py` maps FX nodes to SSA values. Tensors use `!tvm_ffi.tensor` and
retain reference semantics. Graph traversal binds call results through one path;
call dispatch handles scalar operations, ATen and higher-order operators.
The current-node dispatcher delegates ATen, cond, kernel and scalar construction
to dedicated methods; ATen construction owns schema selection and argument
conversion. Checked-call helpers return complete Results; callers bind them
explicitly. Private graph functions return Results; the public wrapper binds them and
uses `tvm_ffi.return` at its packed boundary.
An in-place ATen call operates on the original object, and views keep their
storage aliases. ATen calls use the schema to
order positional and keyword arguments, fill defaults, select a scalar overload
when FX recorded a Python scalar in a tensor overload, and name the existing
`trident.aten.<operator>[.<overload>]` registry wrapper. Dtype arguments pass
through `trident.runtime.tvm_ffi_to_torch_type` to obtain the integer expected by
ATen's boxed schema. Multiple ATen results are retrieved from an FFI Array.

SymInt, SymFloat and SymBool computations use native `i64`, `f64` and `i1` SSA.
Input tensor dimensions bind symbols to `tvm_ffi.tensor.size`; explicit scalar
nodes bind their expressions to imported SSA values. Composite shape expressions
are evaluated from those bindings. Runtime expressions never use example hints;
unbound symbols produce an explicit import error. Native scalars are boxed only
at FFI calls, containers or function return boundaries. `torch.cond` imports its
branches into `scf.if`, with native predicates and monadic Result values.

`ir.py` shares scalar arithmetic, constants, boxing and Result construction with
the guard frontend. `make_result_types` wraps payload types and supplies the
None default for empty sequences. `pure` infers Result types from payloads unless
the caller declares a boundary type; at typed boundaries it boxes native scalars
and widens semantic values to Any/Union before wrapping them. Empty successful
sequences produce one explicit None payload. Shared `checked_call` and
`container_get` builders construct registry calls and typed container extraction
without depending on the graph importer. The shared binary operation mappings
also define the supported FX scalar operations; the importer only normalizes symbolic min/max
aliases.
Each asserting FFI call checks lookup success, a non-null handle and call success;
a failed assertion terminates execution. Generic failed Results still propagate
a nonzero packed ABI status unless handled explicitly at a graph-call boundary. A missing cached global establishes a RuntimeError
on the calling thread; resolved globals retain their cached fast path. Integer
division uses floor semantics; integer remainder follows
`a - floor(a / b) * b`. Supported arithmetic follows the runtime's 64-bit scalar
ABI. Unsupported scalar expressions or higher-order operators fail explicitly.

TVM FFI types have concrete Python bindings exported by
`trident.core.dialects.tvm_ffi`. Their TypeID registrations automatically cast
parsed types and operation value types to classes such as `BoolType`, `IntType`
and `TensorType`. Python type inspection uses `isinstance`, integer width and
signedness, or MLIR Type equality, rather than textual spellings. Native scalar
boxing remains limited to signless i1/i64 and f64. Internal TVM FFI type
construction uses `.get()`; `UnionType.get()` verifies members and exposes them
through its read-only `types` tuple. Union members are restricted to
`TVMFFICABIType`; its C++ parameter and accessor use
`ArrayRef<TVMFFICABIType>`. Native scalars, native Object handles and Results
are rejected during parsing and checked C/Python construction. Existing minimum
arity, uniqueness and indexed-member restrictions remain in force. `ffi_type(name)` remains a textual
compatibility constructor. Only the native C ABI Array uses the `c.` prefix:
`!tvm_ffi.c.array`,
with the `CArrayType` binding. Other ABI types retain their ordinary spellings,
for example `ffi_type("int")` and `!tvm_ffi.tensor`. Semantic `ArrayType.get(T)`
and `TupleType.get(types)` construct `!tvm_ffi.array<T>` and
`!tvm_ffi.tuple<T0, T1, ...>`, including empty tuples. Their read-only
`element_type` and `types` properties expose the parameters. Both semantic
container types inherit `TVMFFICOpaqueABIType`; Array and Tuple elements must be
`TVMFFICABIType`, including nested Array and Tuple types. Array's C++ parameter
and accessor use `TVMFFICABIType`; Tuple uses `ArrayRef<TVMFFICABIType>`.
Parsing and checked C/Python construction reject other types.
C ABI types are identified by the C++ `TVMFFICABIType` class and its
`TVMFFICABI` trait, including indexed object types. Its TableGen subclasses
`TVMFFICBaseABI_Type` and `TVMFFICOpaqueABI_Type` distinguish public C
representations/layouts from opaque semantic representations. The C++
`TVMFFICBaseABIType` and `TVMFFICOpaqueABIType` classes use corresponding traits
for `isa`/`dyn_cast`, independently of object classification. The common CABI
class provides `getLLVMType(context)` for the tagged `{i32, i32, i64}` ABI
representation used by both categories.
Base implements `TVMFFITypeIndexInterface` and supports both fixed and dynamically
registered type indices. Fixed-index POD and object types share the TableGen-only
`TVMFFIStaticTypeIndex_Type` base, which adds `HasStaticTypeIndex` and generates
`static constexpr int32_t getTypeIndex()` from its non-negative `typeIndex`
parameter. The same accessor serves the common interface and compile-time
queries. No additional C++ classification class is needed.
`TVMFFICBaseABI_Type` declares `getTypeIndex()` without an implementation;
the static-index subclass replaces that declaration with its constant
implementation.
Exception implements the declared method in C++ using its runtime-registered
index and does not have the static-index trait.
The C++ Base classification provides an instance `getTypeIndex()` that dispatches
through the interface. The common CABI base and Opaque do not provide that method.
The `HasObject` trait marks CArray, Exception, Function, Str and Tensor
independently of the ABI hierarchy and static-index classification.
The fixed-index object types explicitly declare `TVMFFINativeTypeInterface`;
Exception retains its runtime-registered index. Concrete object types inherit
`TVMFFICBaseABIType` directly. `TVMFFIObjectType` remains a trait-based
classification view supporting `isa`/`dyn_cast` and type-index queries, rather
than a C++ base of concrete types; implicit conversions to that view require an
explicit MLIR cast. `HasObject` identifies definite C ABI object types, not all
reference-bearing types; reference management is selected by the cleanup pass.
`TVMFFI_MayHaveObject` is a TableGen type constraint combining `HasObject` or
`Contains`, rather than an explicitly registered trait. `HasObject` directly
inherits the MLIR type-trait base. Any and Union register only `Contains`;
Result and native Object register neither marker. After unwrapping a Result
payload, the cleanup pass checks native Object or either trait. Scalar Results
need no references, while even scalar-only Unions conservatively require reference
handling. Semantic Array/Tuple are classified after conversion to CArray.
These rules preserve ABI classification, representation and cast legality.
Any and Union carry the `Contains` native type trait, which declares
`bool contains(TVMFFICABIType) const`, accepting only C ABI types. Any always
returns true; Union checks direct membership or whether every member of another Union belongs to it.
Any, semantic Array/Tuple and Union belong to Opaque.
Object remains a native-pointer `TVMFFIABIType` outside CABI; it cannot
be a Tuple member, tagged call value or tagged cast operand/result. It retains
reference management and native Result payload support. `decompose-tvm-ffi` converts semantic Array/Tuple to the concrete Base
`CArrayType`, including function signatures, region arguments, Result payloads
and type-bearing attributes. Other opaque types retain their specialized ABI/LLVM
lowering paths; opaque classification alone does not imply container erasure.
Result remains an independent monadic wrapper outside the ABI classification.
All current fixed-index types belong to Base, including CArray (TVMFFISeqCell),
Function (TVMFFIFunctionCell), Str (TVMFFIByteArray) and Tensor (DLTensor).
This classification does not change native extraction or LLVM storage: Function
still lowers to a native pointer.
Trident extensions retain `!tvm_ffi.exception`, `!tvm_ffi.union` and
`!tvm_ffi.monad.result<T>`; their payloads use the ordinary ABI types.

## Guards and containers

Guards are parsed into `GuardCode`, `Guard` and collection objects under
`python/trident/guards/`. `InputTableBuilder` maps function arguments and nested
list/tuple paths to semantic values. Lists use `!tvm_ffi.array<T>` with a common
boxed element type, or `!tvm_ffi.any` for empty or heterogeneous lists. Tuples
retain one type per element. `array_type` infers a common boxed element type;
`tuple_type` boxes and preserves each element type independently. Nested
containers preserve this structure.
The frontend emits `array.create/get/size` and `tuple.construct/unconstruct`.
Array construction and extraction return their array/element types directly;
array size returns a boxed integer directly. Tuple construct directly packs elements into a semantic Tuple;
unconstruct directly returns all elements in order, including zero results for
an empty Tuple. Tuple indexing selects an unconstruct result, normalizes negative
indices and checks static bounds; static length comes from the Tuple type.
Container decomposition checks lookup status, non-null handles and call status with
runtime assertions that terminate on failure. Array indexing uses a
native i64; runtime failures terminate through assertions. `InputTable.extract`
builds and caches complete Results for a path; `extract_many` reconstructs leaf
inputs in exported order. Extraction is lazy in its consumer block. Guards use
these table methods and shared IR builders directly, without a graph importer.
Tuple length guards retain checked runtime `ffi.ArraySize` calls: wrapper inputs
must validate arity before trusting their static tuple schema.
A single input-node implementation handles leaves and containers; each fresh
table keeps a scope-checked borrowed cache of complete extraction Results. Scalar guards
use native arithmetic and comparisons; tensor metadata uses existing semantic
metadata operations. Guard collections build structured short-circuit branches
with fresh input tables per region and explicitly bind fallible extraction
Results. A failed specialization returns `!tvm_ffi.exception` widened to
`!tvm_ffi.any`, so the dispatcher can try another specialization or request
compilation.

Tensor identity and `_base` guards retain their existing conservative handling;
DLPack does not expose Python wrapper identity or autograd view relationships.

## Triton kernels

`importer.py` resolves the compiled Triton kernel from its warmup cache,
imports its cubin as `gpu.binary`, and emits `tvm_ffi.kernel_launch`.
The kernel construction method resolves the cache, builds specialization
attributes and assembles operands. Recursive launch construction clones functional
outputs, inserts the binary and emits the launch, returning complete Results
for its output tensors.
Each specialization has a unique binary symbol. Operands retain semantic FFI
values and carry constant or variable specialization attributes in source
parameter order. Variable attributes record native ABI type and divisibility;
constant attributes include scalar, string and nested tuple values. Binder hints
are used only to select the compiled kernel, never to substitute runtime shapes.
Functional Triton wrapper outputs explicitly clone `tensors_to_clone` before
launching; mutation wrappers use the original operands.

`tvm_ffi.kernel_launch` returns `Result<None>`. `ConvertTVMFFIToGPU`
validates specializations and emits `gpu.launch_func` with the current TVM FFI
CUDA stream only on a match. That branch returns successful None; rejection
returns failure without setting a runtime error or exiting the function.
Constant operands are checked and omitted from the native argument list.
Python binds the launch result, skipping later computation on rejection.
Success means the launch path was taken, not asynchronous execution completed.

## Lowering and ownership

The lowering pipeline is:

1. Run the TVM FFI dialect transform `decompose-tvm-ffi` to expand container
   operations into checked `ffi.Array`, `ffi.ArrayGetItem` and `ffi.ArraySize`
   calls and lower monadic binds to `monad.unpack` and `scf.if` in one dialect
   conversion. The same conversion erases container types to `!tvm_ffi.c.array`
   in function signatures, region arguments and Result payloads, using standard
   function and control-flow patterns plus explicit TVM FFI patterns. Bind
   continuations move into success branches with converted block arguments;
   their yields use remapped operands. Unsupported operations carrying container
   types fail conversion; nested type attributes are erased separately with
   `AttrTypeReplacer`. Results remain single semantic SSA values. Independent
   references produced by container construction and extraction receive
   `tvm_ffi.cleanup`; borrowed casts and arguments remain unmarked.
2. Run `materialize-tvm-ffi-cleanup` on this structured CArray IR. It consumes
   explicit cleanup markers and `tvm_ffi.retain_payloads` on frontend pure Ops,
   retaining each borrowed payload occurrence before wrapping it and retaining
   exports before local releases. Result reference adjustments directly use
   `monad.unpack` and independent `scf.if` success branches. Inline private
   graph functions and remove unused symbols only after cleanup, then lower SCF
   to CFG.
3. Lower kernel launches to GPU operations and finalize metadata operations.
4. Preserve the reference operations emitted by cleanup for borrowed payloads
   and exports. Each lowering releases its own temporary resources,
   including function handles introduced during container decomposition.
5. Convert monadic returns to checked functions with output slots and an i32
   status, and generate packed ABI wrappers. Read Results through `monad.unpack`
   and restore typed payloads only in the successful branch before storing them.
   Native Function/Object payloads can use internal output slots, but packed
   wrappers require tagged ABI payloads.
6. Lower semantic values, including Results, to LLVM. A `Result<T>` becomes one
   LLVM struct containing the converted payload followed by an i1 success bit.
   Lower DLPack, arithmetic, control flow, GPU and UB operations, followed by
   canonicalization and cast reconciliation.

`!tvm_ffi.monad.result<T>` always carries one payload and a success bit.
No-data results use `!tvm_ffi.monad.result<!tvm_ffi.none>` with an explicit
`tvm_ffi.constant.none` payload. `monad.pure` constructs successful
Results, and `monad.fail` constructs failure without replacing the runtime
error. `monad.bind` runs its continuation only when every input succeeds;
its body terminates with `monad.yield`. Input and output counts are independent
and may be zero: no inputs runs the body unconditionally, and no outputs performs
only the body's effects when inputs succeed. Decomposition moves that body into
the success branch of `scf.if`, restores typed payloads there with
`tvm_ffi.downcast` when unpack widens them to Any, and yields failed Results in
the failure branch when there are outputs. A zero-output bind needs no failure
branch or failed Results.
Decomposition handles binds throughout the module, including ordinary helper
functions. LLVM conversion lowers Result types and `monad.pure`, `monad.fail`
and `monad.unpack` together, preserving SSA value counts through ordinary
function signatures, calls, returns and CFG arguments. The default pipeline
inlines graph helpers after decomposition and cleanup. Unsupported Result uses fail conversion.
Redundant Any round trips remain a TODO; they do not change failure propagation.
Independent effects after a checked call are also nested in its continuation,
so failure suppresses them without requiring a payload dependency.

`TridentFxImporter.visit` consumes a shared iterator over the original FX nodes.
Pure nodes stay in a loop. Node handlers construct only their current node and
return its ordinary value structure, whose leaves may include complete Results.
The visitor flattens
this structure, binds its Result leaves, restores payloads in place, and
recursively imports the remaining iterator inside the successful body. Unit
Result leaves with a None payload restore to Python `None`, consuming their
corresponding block argument. The visitor does not redispatch on the node target
after binding. Each bind yields inner
Results as the stack unwinds. Final outputs use `pure`, including a None
payload for no-data returns.

Ordinary scalar, schema and metadata helpers return ordinary values. Independent
fallible helpers construct arguments, checked calls, ATen conversions and result
extraction, inputs, or guard expressions and return complete Results. One
argument builder automatically converts Python `torch.dtype` arguments to ATen's
integer enum, including elements of nested lists and tuples. Existing SSA values
keep their types; ordinary dtype constants and guard expressions retain the TVM
FFI dtype representation.
Ordered argument, input, guard-operand and result extraction builders carry
accumulated payloads into each successful bind and wrap them once at the end.
They keep pure prefixes iterative and nested list/tuple elements recursive,
preserving element order and failure short circuiting.
Empty argument and input builders return one None Result to preserve success
tracking. Consumers use the original list length to omit this placeholder from
actual call arguments; explicit Python None arguments retain their positions.
Argument construction does not invoke an operator or traverse later FX nodes. Kernel clone, grid and
launch construction likewise preserves its original order.

The shared `trident.ir.bind` construction context manager borrows its inputs.
The Python builder unconditionally marks the bind operation for block-exit cleanup;
the C++ cleanup pass selects its reference-bearing results. Successful body arguments remain borrowed. It
yields the `MonadBindOp`; its read-only `arguments` property exposes
one successful payload per input in input order, including None payloads,
while `outputs` exposes complete Results.
The property is also available on parsed and traversed OpViews. MLIR verification
checks the successful terminator and yielded Result types. The context manager
gets its body through `MonadBindOp.add_entry_block()`, which infers argument
types from input Result payloads; C++ builders use
`MonadBindOp::addEntryBlock(builder)`, which creates the block through the builder
and sets its insertion point to the new block when the body is empty. Both
methods return the existing entry block without changing it when present and
leave terminators to the caller. Retrieving an existing block preserves the
builder insertion point. The Python method always preserves the insertion point
and accepts optional argument locations used only when creating the block.
`getEntryBlock()` and Python `get_entry_block()` only query the body, returning
null or None when no entry block exists without allocating or changing insertion
points.
The context manager
restores insertion points on normal and exceptional exits. There are no
continuation callbacks, Effects objects, generator resume protocols or task
queues. Deep fallible chains may exceed the unchanged Python recursion limit;
pure FX nodes and pure argument prefixes do not add recursive frames.
`import_program` and direct `visit` calls retain their value and symbol mappings;
use a fresh importer for each program.

Python imports `CLEANUP_ATTR` and `RETAIN_ATTR` from `trident.ownership`;
these constants name `tvm_ffi.cleanup` and `tvm_ffi.retain`.

Frontend exits construct `func.ReturnOp`, `tvm_ffi_d.ReturnOp`,
`scf.YieldOp` or `tvm_ffi_d.MonadYieldOp`, then directly set
`operation.attributes[RETAIN_ATTR] = ir.UnitAttr.get()`.
The cleanup pass consumes this UnitAttr, retaining each reference-bearing
operand occurrence before releasing local references. Unmarked exits request
no automatic retains. Do not manually retain outputs for marked exits.
Payload retains requested by `pure` remain independent of export retains.

Function entries and branches use insertion points directly. Cond branches use
independent importers sharing the parent module, and return Results for the
parent visitor to bind. Input tables cache complete Results, never extraction
bind payloads. Cached Results must belong to an ancestor block; sibling scopes
rebuild their extraction. Input and guard tables keep cached SSA Results until
discarded or cleared; resource reuse also requires an explicit cleanup marker on the cached
reference. Ownership lasts until the defining IR block exits, independent of
Python context-manager exits. Guards and backend input flattening use fresh
tables, so caches remain local to each construction.
Construction exceptions propagate without automatic cache clearing; discard the
importer and its input tables after a construction failure rather than reusing
them. A guard consuming an extracted value receives a separate structural table
scoped to its bind. Imported payloads check block ancestry before reuse.

Each guard-expression visitor caches one recursive analysis per AST node for its
type, active operands, purity, fallback and checked array-size requirement.
Synthesized operands are reused, while expression values are evaluated afresh;
comparison chains still evaluate their intermediate operands twice. Both pure
and fallible construction reuse the cached active operand list; comparison
construction slices its operand pairs from that list and uses the cached
whole-expression fallback instead of rescanning the remaining chain. Analysis
stops at fallback so later operands are neither resolved nor built. Guard
expressions share ordinary scalar evaluation helpers and consume pure operand
prefixes in a loop before opening checked-operation binds. Fallible expression
helpers return complete Results; static fallback metadata preserves
whole-expression `_base` and storage-offset behavior without a continuation
protocol. Previously constructed calls still run and enforce their assertions;
fallback suppresses later operands. Guard collections return a short-circuiting
boolean Result. The backend explicitly binds this result, constructs the guarded
branch, binds flattened inputs and the graph call, then packs the output into
the public Result. Each `tvm_ffi.func` declares exactly one Result return type, and
`tvm_ffi.return` must match the signature exactly. No-data functions declare
`Result<!tvm_ffi.none>`; these results use the ordinary TVM FFI None ABI
representation and allocate output slots just like other payloads.
Successful payloads are wrapped with `monad.pure`; the shared Python `pure`
builder performs boxing and Any/Union widening at declared boundaries. Private
graph helpers use `func.return` with Result values and are inlined before Result
lowering. Packed output slots are read and written only on success. Packed
wrappers forward the single ABI payload directly. Multiple values must be
packed into a `!tvm_ffi.tuple` payload before returning; tuple decomposition
constructs the runtime Array before ABI conversion. Guard rejection and Triton
specialization failure remain successful `GuardMatch` values, requesting
another specialization; runtime FFI failures return nonzero status without
inspecting output payloads.

Failed semantic ABI payloads are zero-initialized. Function handles use null
native-pointer placeholders; native scalar payloads may be poison and are never
consumed on failure. No Result object is allocated at runtime. Result retain and
release use `monad.unpack` and independent `scf.if` success branches, accessing
payloads only when their own Results succeed. Tagged payloads widened to Any
are adjusted directly; native handles retain their types. These branches carry
no values or additional references through their yields.

The TVM FFI dialect distinguishes boxed values, owned runtime objects, borrowed
objects and native pointers. Reference-bearing type selection is private to the
cleanup pass: it unwraps Result payloads and checks native Object, HasObject or
Contains, conservatively including Any and Union. Python and decomposition
record cleanup intent without classifying result types; the cleanup pass selects
reference-bearing results after semantic container decomposition. Ownership
interfaces describe operation contracts. Container runtime calls and converted
FunctionCall operations acquire cleanup markers unconditionally during decomposition; other
resource-producing operations require explicit cleanup intent. The explicit
`tvm_ffi.cleanup` UnitAttr schedules one release for every
reference-bearing result of an operation, including each resource in mixed
multi-results. Scalars are ignored. Unmarked results remain unmanaged. Cleanup
responsibility describes object references, even when they share tensor storage.

`materialize-tvm-ffi-cleanup` assumes well-formed input attributes and processes
operations only when their immediate parent region has one block. Three pattern
walks materialize borrowed pure payload retains, then export retains, then
consume cleanup markers at block exits. The walks do not fold or eliminate operations, and cleanup patterns
explicitly preserve reverse operation and result
order independently of traversal order. Region selection does not use an operation
whitelist. Operations directly in multi-block regions retain their attributes
and are skipped; nested single-block regions can be processed independently.
The walk driver does not visit unreachable blocks.
Results of the region-owning operation are still processed in its selected parent
block. Export retains are generated for marked `func.return`, `tvm_ffi.return`,
`scf.yield` and `tvm_ffi.monad.yield`.
Each export is retained in operand order before any local release. References
are released in reverse operation and result order. Result payloads are inspected
only on success, including partially successful multi-results.
Consumed cleanup and `tvm_ffi.retain` markers are removed, including
export markers on scalar-only and empty exits. This makes repeated cleanup runs
idempotent. Container and bind decomposition preserve pending markers on
converted function returns and region yields. Generated failure exits carry no
references and need no export marker.

Python builders construct the operation, then directly set
`operation.attributes[CLEANUP_ATTR] = ir.UnitAttr.get()` for acquired
results. Marking an operation covers all its reference-bearing results;
repeated assignment does not add credits. Scalar-only and zero-result operations
can carry the marker; the C++ pass consumes it without emitting releases.
There are no marking helper functions or automatic annotation stage. Cache
checks require lexical visibility and an OpResult whose defining operation has
a cleanup UnitAttr, regardless of result type. Marked terminator exports acquire
their references during cleanup.

Resource helpers borrow arguments and explicitly mark their acquired results
for cleanup. `pure` borrows payloads and records `tvm_ffi.retain_payloads` to
request an independent reference for each resource Result after container erasure,
including repeated payloads; callers do not retain them again.
The low-level `MonadPureOp` transfers references, so callers must manage ownership
explicitly. The cleanup pass retains each marked exported occurrence for
the receiving scope. Block arguments, captures, casts and unpacked payloads borrow
their source. ATen views can own references while sharing storage. Lowerings
preserve cleanup markers on replacement resource operations. Temporary references
with explicit releases remain unmarked to avoid duplicate release.

Construction failures abandon incomplete IR; discard the importer and tables.
The first version does not perform early-release optimization or full borrow
checking: SSA visibility alone does not establish lifetime validity.

Generated private graph Results use failure for specialization rejection. At the
Python-built graph-call boundary, the wrapper checks all Result success bits,
reads/packs payloads only on success, and maps rejection to a successful GuardMatch
payload. All independently successful outputs are cleaned up even if another
output failed. Ordinary asserting FFI-call failures remain assertions, and generic
Result/ABI failure propagation is unchanged. Introducing recoverable graph runtime
errors requires distinguishing them from rejection before extending this contract.
Runtime objects use `!llvm.struct<(i32, i32, i64)>`; native Object/Function handles
are pointers. `ResultType` retains its C/Python bindings and TypeID registration.

`tvm_ffi.upcast` widens a tagged semantic ABI value to Any or a compatible union;
`tvm_ffi.downcast` narrows Any/Union to a concrete type or a smaller Union,
checking the result type against the runtime type tag. Failed checks branch to
an `abort()` call and LLVM unreachable in every build mode. A shared LLVM assertion helper
contains a TODO for diagnostic output before aborting. Semantic containers support Any widening/narrowing; after erasure, narrowing
checks only the Array runtime tag, without validating element types. Neither cast supports
native Function/Object handles or unions containing them. Both casts borrow
their source; downcast checks remain observable even when its result is unused.
`tvm_ffi.get` extracts a native scalar or borrowed object, and `tvm_ffi.to` boxes
a native scalar. Metadata operations lower through DLPack. Tensor literals use
the existing runtime staging path. Global function handles are cached at module
initialization; call sites retain and release the cached reference.

`tvm_ffi.monad.unpack` exposes one Result as a success `i1` followed by its
payload. Payload types compatible with `tvm_ffi.upcast` to Any produce
`!tvm_ffi.any`: success preserves the value and failure produces None.
Other payload types, including native scalars and Function/Object handles,
remain unchanged and may only be used when success is true. Unpack neither
clears runtime errors nor transfers ownership. Result payload types remain
unrestricted except for the existing prohibition on nested Results.

## Source layout and validation

`core/` contains the DLPack and TVM FFI dialects, lowering passes,
C API and Python bindings. `ffi/` generates ATen wrappers from dispatcher schemas
and provides runtime conversion and exception helpers. Python guard parsing and
expression IR and importer diagnostic tests live under `test/unit`; public JIT tests live under
`test/e2e`. Dialect, conversion and pipeline tests live under `core/test/`. See
[the Python test guide](../test/README.md) for coverage boundaries.
Operator work follows [Operator-Adaptation.md](Operator-Adaptation.md).

Build with the editable command in the repository contributor guide, then run
`python -m unittest discover -s test -p 'test_*.py'` and
`cmake --build build/core --target check-trident-core`. A standalone core build
uses an installed MLIR CMake package; it does not fetch another compiler frontend.
CUDA is required by GPU execution tests.

The core and FFI external projects always run their incremental build during a
top-level build or editable install. This keeps installed native extensions in
sync with source changes even when ExternalProject configure stamps are current.
