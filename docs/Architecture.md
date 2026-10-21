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
the guard frontend. `make_result_types` boxes native scalar types before wrapping
payload types and supplies the None default for empty sequences. `pure` infers CABI Result types
from payloads unless the caller declares a boundary type; it boxes native
scalars before wrapping them and widens semantic values at Any/Union boundaries.
Bind arguments retain their CABI types; consumers extract native scalars for
arithmetic and boolean conditions. Empty successful sequences produce one explicit None payload. Shared `checked_call` and
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
`trident.core.dialects.tvm_ffi`, with TypeID registration for parsed types and
operation value types. Inspect them through `isinstance`, parameter accessors,
MLIR Type equality, and native integer width/signedness rather than text.
`ffi_type(name)` remains a textual compatibility constructor.

The Python frontend constructs only object-free Base types and Opaque types.
Monad `ResultType` remains an independent wrapper whose payload is CABI.
Object-bearing Base types have `C`-prefixed binding names and `c.` mnemonics:
`CArrayType`, `CExceptionType`, `CFunctionType`, `CStrType` and `CTensorType`.
Their semantic counterparts are Array/Tuple, Exception, Function, Str and Tensor.
Low-level bindings remain available for inspection and tests; this is a frontend
IR construction contract, not a restriction on MLIR parsing or Python `.get()`.
Native `ObjectType` is a pointer outside CABI and is generated only by C++
lowering, not by the Python frontend. It cannot be a Result or container payload.
Global function lookup returns semantic Function, and registry calls accept it
as their callee; decomposition changes both to CFunction. Textual registry calls
explicitly include their callee type.

`TVMFFICABIType` and its trait classify all tagged C ABI types. Base types have
public C representations; indexed Base types implement `TVMFFITypeIndexInterface`.
CAny has a dynamic tag and no single type index. Opaque types expose
semantic representations requiring decomposition. The common tagged layout is
`{i32, i32, i64}`, while CFunction retains its native-pointer LLVM storage and is
boxed at tagged ABI boundaries. Base POD and fixed-index object types use the
TableGen-only `TVMFFIStaticTypeIndex_Type` to generate a constexpr index accessor.
CException supplies its runtime-registered index instead. Opaque types do not
expose runtime type indices.

The Opaque/Base separation serves two design goals:

1. Automatically select an appropriate underlying implementation while keeping
   frontend IR expressed in semantic types. Opaque-to-Base conversion is the
   boundary where lowering chooses concrete representations and operations;
   frontend users should not need to make that choice.
2. Ultimately automate memory management. The same conversion boundary exposes
   the concrete representation and its ownership requirements so lowering can
   materialize the necessary lifetime management.

Opaque-to-Base conversion selects concrete representations and materializes
semantic reference lifetimes. Python builders do not attach ownership attributes.
The automatic policy applies to semantic objects; handwritten Base operations
and their explicit reference adjustments remain manually managed.

The `HasObject` trait marks precisely the five object-bearing Base types.
`TVMFFIObjectType` is a trait-based classification view of those types, not their
concrete C++ base. Any, CAny and Union carry `Contains`; Any and CAny contain all CABI types,
while Union checks direct or subset membership. Ownership materialization uses
the original Opaque types at each operation and region boundary. Base-typed results remain manually managed, including elements
extracted from semantic containers. Any/Union reference adjustments check the
runtime tag after lowering.
Object-free Base types remain Bool, DType, Device, Float, Int, None, RawStr and
SmallStr. A scalar-only Union conservatively receives runtime reference checks.

Str is a unified semantic string, including raw views, inline small strings and
heap objects. `constant.str` creates a semantic string; decomposition produces a
RawStr constant widened to CAny. This preserves the
owning-container normalization of raw views to SmallStr or CStr. Result payloads,
container elements and frontend string inputs all use Str. Str and Union become
CAny, even when Union members share one representation. Preserving their finite
runtime tag sets for downcast checks is deferred.

`ArrayType.get(T)` and `TupleType.get(types)` require CABI elements and preserve
nested semantic containers. They expose `element_type` and `types`; empty tuples
are supported. Union requires at least two unique CABI members with runtime
indices or supported semantic object representations, excluding Any, CAny and nested
Unions. Parsing and checked C/Python construction reject invalid members and
cross-context construction. Result likewise requires one CABI payload.

`decompose-tvm-ffi` converts Array/Tuple to CArray, Exception to CException,
Function to CFunction, Tensor to CTensor, and Any/Union/Str to CAny.
It recursively converts Result payloads, signatures, region
arguments and type-bearing attributes. It also inserts semantic export retains
and registers local releases in `tvm_ffi.defer` regions. CAny retains the tagged
LLVM layout. Original semantic types determine ownership; converted Base types
make operations legal without a separate processed-operation set.
Finite Union/Str constraints currently disappear when converted to CAny, which
accepts every tag. Preserving these constraints for explicit downcasts, packed
argument validation and success-only return checks is deferred. Downcasts to
concrete indexed Base types retain their runtime checks.
Frontend builders do not classify references themselves.

Type definitions and implementations are organized under TVM FFI IR `Base/`,
`Opaque/` and `Monad/`. Monad contains Result and all `monad.*` definitions and
implementations. Top-level files retain common ABI utilities, native Object,
aggregate constraints and registration. `TVMFFITypeBase.td` and
`TVMFFIOpBase.td` provide the shared TableGen hierarchy and operation base.

## Guards and containers

Guards are parsed into `GuardCode`, `Guard` and collection objects under
`python/trident/guards/`. `InputTableBuilder` maps function arguments and nested
list/tuple paths to semantic values. Lists use `!tvm_ffi.array<T>` with a common
boxed element type, or `!tvm_ffi.any` for empty or heterogeneous lists. Tuples
retain one type per element. `array_type` infers a common boxed element type;
`tuple_type` boxes and preserves each element type independently. Nested
containers preserve this structure. Strings use semantic Str throughout the frontend, including Result payloads
and nested container elements. Decomposition admits RawStr, SmallStr and CStr
representations after runtime container normalization.
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

1. Lower kernel launches to GPU operations, retaining their success/rejection
   Results.
2. Run `lower-tvm-ffi-monad`. Every `Result<T>` becomes a semantic
   `tuple<bool, any>`. Pure values become `(true, payload)` and failures become
   `(false, None)`. Binds become tuple extraction and `scf.if`; their typed
   payloads are restored only in successful branches. Function signatures,
   calls and control-flow arguments are converted together.
3. Run `decompose-tvm-ffi`. Preserve local `tvm_ffi.call` operations and
   convert semantic object types to their Base representations throughout
   signatures, regions and type-bearing attributes, including Array/Tuple to
   `!tvm_ffi.c.array` and Any/Union/Str to `!tvm_ffi.c.any`. Container operations become
   checked runtime Array calls. Materialize semantic export retains and register
   local releases in defer regions. Run `lower-tvm-ffi-defer` immediately
   afterward to move their bodies
   to block exits before inlining and SCF-to-CFG conversion.
4. Finalize metadata operations and convert TVM FFI functions and returns to
   checked functions with output slots and an i32 status, generating packed ABI
   wrappers. Output slots are read and written only on success.
5. Run GPU-to-LLVM before expanding packed calls and reference checks, avoiding
   region simplification over the expanded LLVM CFG. The GPU conversion also
   applies registered dialect LLVM patterns. Expand remaining local packed calls
   in `TVMFFIToLLVM`, then convert semantic ABI values
   and the remaining dialects to LLVM, followed by
   canonicalization and cast reconciliation. No Result types reach LLVM lowering.

`tvm_ffi.monad.call @callee(...)` returns `Result<T>`, where T exactly matches
its callee's return type. `lower-tvm-ffi-monad` emits `tvm_ffi.call @callee(...)`
with separate `!tvm_ffi.bool` and `!tvm_ffi.any` results, then explicitly packs
them into the semantic Bool/Any tuple used for lowered Results. The tuple
borrows the call payload and retains it through runtime container construction.
Decomposition releases both the call's independent reference and its local tuple
reference at block exit, retaining any exported tuple first.
Result attributes move to the call's payload result, with an empty entry for success.
Local callee resolution, argument matching and the `emit_tvm_ffi_abi`
requirement are deferred to lowering for unlowered callees. This local-symbol
call is distinct from registry
`tvm_ffi.FunctionCall`.

`decompose-tvm-ffi` preserves the call's two results while converting container
operands and explicitly constructed tuples to CArray. `TVMFFIToFunc` converts
function definitions and returns, preserving the implementation's
`emit_tvm_ffi_abi` marker for local call verification; it does not expand calls.

`TVMFFIToLLVM` requires decomposed container operands and expands the call through
`__tvm_ffi_<callee>` using packed argument and output slots. It emits a direct
`func.call` to the wrapper supplied by `TVMFFIToFunc`; it does not look up,
validate or create wrapper declarations. Zero status produces separate
`(true, payload)` results; nonzero status produces `(false, None)` without reading
the output slot or clearing the runtime error. The payload owns the packed
return reference. Existing payload retains and releases follow SSA replacement;
the Bool result needs no reference management. Local call lowering performs no
runtime Array construction, registry lookup or temporary reference adjustment.
Only consumers requiring a container explicitly construct one before cleanup.

`!tvm_ffi.monad.result<T>` carries one CABI payload and success state before
monad lowering. No-data Results use an explicit None payload. `monad.bind`
runs its continuation only when every input succeeds; input and output counts
are independent and may be zero. Zero inputs execute unconditionally; zero
outputs perform only conditional effects. A failed bind yields failed Results
when outputs exist. Independent effects after a call remain inside its
continuation, so failure suppresses them without requiring a payload dependency.
The existing `func.call` path for private graph helpers remains separate from
local packed calls. Function payloads are boxed into Any during monad lowering.

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
Decomposition manages the bind's lowered Result containers automatically.
Successful body arguments remain borrowed. It
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

Python builds ordinary operations and exits without ownership attributes.
Function entries and branches use insertion points directly. Cond branches use
independent importers sharing the parent module, and return Results for the
parent visitor to bind. Input tables cache complete Results, never extraction
bind payloads. Cached Results must belong to an ancestor block; sibling scopes
rebuild their extraction. Semantic reference lifetimes extend to the defining
IR block exit, independently of Python context-manager exits. Guards and backend
input flattening use fresh tables. Construction exceptions propagate; discard
an importer and its tables after a construction failure.

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
boxed boolean Result. The backend explicitly binds this result, constructs the
guarded branch, binds flattened inputs and the graph call, then packs the output into
the public result. Each `tvm_ffi.func` declares exactly one C ABI payload type;
`tvm_ffi.return` carries a boxed success boolean and Any payload. No-data
functions declare `!tvm_ffi.none`. Private graph helpers use `func.return` with
Result values, which monad lowering converts to tuples before cleanup and
inlining. Packed output slots are read and written only on success. Multiple
values are packed into a semantic tuple payload before returning. Guard
rejection and Triton specialization failure remain successful `GuardMatch`
values, requesting another specialization; runtime FFI failures return nonzero
status without inspecting output payloads.

Lowered Results are runtime Array tuples containing a boxed boolean and payload.
Failure tuples contain None. Both the Python `pure` helper and low-level
`MonadPureOp` borrow their inputs: tuple construction retains each occurrence,
including repeated payloads. No transfer mode or ownership attribute is needed.

`decompose-tvm-ffi` validates lexical ownership scopes without tracing value
provenance or collecting ownership actions. `ReplacementConversionPattern<SourceOp>`
derives from `OpConversionPattern` and provides protected static replacement
helpers. Producers explicitly select its `OwnershipConversionPattern<SourceOp>`
subclass, whose helpers register local releases for original Opaque result types
in a `tvm_ffi.defer`, in reverse result order, before delegating replacement.
Representation-only patterns select their base through a template parameter;
there is no runtime producer classification in the replacement helpers.
Scope validation checks region structure, while conversion legality depends on
types. Casts, parameters and captures borrow and create no local cleanup
responsibility. Tensor storage aliases do not imply borrowed object references.

Each return or yield pattern retains every Opaque operand in operand order,
including repeated values, borrowed parameters and values widened from Base or
POD types. The operand's source does not affect this export rule. Each `scf.if`
Opaque result receives an independent reference from its selected branch and
registers a release in its enclosing block. Nested regions use the same rule
without propagating ownership through Base-valued results.

The decomposition implementation separates representation mapping, container
runtime calls, ownership completion and conversion configuration. Array and Tuple
construction share one pattern; container patterns explicitly release their
temporary Base function handles after each runtime call. Producer patterns inherit
the ownership-aware conversion pattern base and use its replacement helpers,
which register semantic cleanup through the driver-provided rewriter in the same
conversion transaction. Value replacements register cleanup at the current
insertion point; operation replacements place it
after the replacement operation. Dedicated return/yield
patterns retain exported operands without applying producer ownership rules.
The pass validates scopes, converts operations, then recursively converts
type-bearing attributes.

Patterns use original types to classify ownership and remapped operands and
replacement results to emit reference operations. Any/Union become CAny, so
type legality forces their ownership patterns to run once without tracking
completed replacements. Patterns run before conversion-time folding so
a producer or branch cannot vanish before registering its ownership. Region
conversion moves existing child operations rather than cloning them.

`lower-tvm-ffi-defer` moves cleanup bodies before the registering block's
terminator, after all export retains. Each enclosing region must be single-block.
Inner lexical scopes run before outer scopes; registrations in the same block
run in reverse order, preserving operation order and SSA data flow within each
body. Runtime-tagged reference checks
handle CAny payloads, including scalar-only values. Temporary Base
function handles created by decomposition keep their immediate, explicit
releases; generated runtime calls are not reclassified as semantic producers.
Original Base results remain manually managed, including Base-typed container
extractions. A manually managed reference exposed through an Opaque return or
yield still receives the independent export reference required by that boundary;
its original reference remains the caller's responsibility to manage explicitly.

Automatic management requires single-block regions and currently supports
functions and `scf.if`. Required ownership across multi-block CFG edges or
unsupported region exits is diagnosed rather than silently skipped. The dedicated
`lower-tvm-ffi-defer` cleanup pass runs immediately after decomposition. Each
defer has a single block without arguments, captures its inputs, and ends in
`tvm_ffi.defer_yield`. Nested regions and additional terminators in cleanup
bodies are rejected. Automatically generated cleanup stays within its
registering block, including `scf.if` branches; it does not unwind aborts or
exceptions. Defer lowering supports only single-block `tvm_ffi.func`, `func.func`
and `scf.if` regions with `tvm_ffi.return`, `func.return` or `scf.yield` exits.
Multi-block regions, unsupported enclosing operations and unsupported exits
produce diagnostics before any cleanup is moved.
Decomposition is a once-only lowering boundary and does not promise idempotent
reference materialization. Defer lowering is a no-op when no defers remain.
There is no early-release optimization or complete borrow checker; callers must
preserve lexical validity.

Generated private graph Results use failure for specialization rejection. At the
Python-built graph-call boundary, the wrapper checks all Result success bits,
reads/packs payloads only on success, and maps rejection to a successful GuardMatch
payload. All independently successful outputs are cleaned up even if another
output failed. Ordinary asserting FFI-call failures remain assertions, and generic
Result/ABI failure propagation is unchanged. Introducing recoverable graph runtime
errors requires distinguishing them from rejection before extending this contract.
Runtime objects use `!llvm.struct<(i32, i32, i64)>`; native Object/CFunction handles
are pointers. `ResultType` retains its C/Python bindings and TypeID registration before monad lowering.

`tvm_ffi.upcast` widens a tagged semantic ABI value to Any or a compatible union.
Both casts accept identical source and result types as a no-op.
`tvm_ffi.downcast` narrows Any/Union to a concrete type or a smaller Union.
After decomposition, concrete indexed Base targets check the runtime type tag;
finite Union/Str checks are deferred. Failed checks branch to
an `abort()` call and LLVM unreachable in every build mode. A shared LLVM assertion helper
contains a TODO for diagnostic output before aborting. Semantic containers support Any widening/narrowing; after erasure, narrowing
checks only the Array runtime tag, without validating element types. Neither cast supports native Object handles. Function values preserve their
existing tagged boxing and native-pointer extraction paths. Both casts borrow
their source; concrete indexed Base downcast checks remain observable even when
the result is unused.
`tvm_ffi.get` extracts a native scalar or borrowed object, and `tvm_ffi.to` boxes
a native scalar. Metadata operations lower through DLPack. Tensor literals use
the existing runtime staging path. Global function handles are cached at module
initialization; call sites retain and release the cached reference.

`tvm_ffi.monad.unpack` exposes one Result as a success `i1` followed by its
payload. Payload types compatible with `tvm_ffi.upcast` to Any produce
`!tvm_ffi.any`: success preserves the value and failure produces None.
Function payloads use the same Any widening, with pointer boxing supplied by
the underlying CFunction lowering. Unpack neither clears runtime errors nor
transfers ownership. Result payloads must be `TVMFFICABIType`; native scalars,
native Object handles and nested Results are rejected by parsing and checked
C/Python construction. The C++ payload parameter and accessor use
`TVMFFICABIType`. Function remains a valid CABI payload with native-pointer
LLVM storage; CABI membership does not imply compatibility with Any casts.

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
