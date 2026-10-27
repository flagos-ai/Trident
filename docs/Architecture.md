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
compiled program. A private `tvm_ffi.func` holds the graph, and a public
`tvm_ffi.func` checks guards and exposes the packed ABI. The Python executor
converts runtime tensors through DLPack and reconstructs the output pytree.
The semantic `tvm_ffi.pack.func` dispatcher tries packed specializations in order with
`tvm_ffi.pack.call`, tests successful payloads with `tvm_ffi.isa`, and propagates
runtime call failures. Automatic ownership releases rejected Exception payloads.
The combined module, including the dispatcher, is lowered in one pipeline.

`importer.py` maps FX nodes to SSA values. Tensors use `!tvm_ffi.tensor` and
retain reference semantics. Graph traversal binds call results through one path;
call dispatch handles scalar operations, ATen and higher-order operators.
The current-node dispatcher delegates ATen, cond, kernel and scalar construction
to dedicated methods; ATen construction owns schema selection and argument
conversion. Checked-call helpers return complete Results; callers bind them
explicitly. Private graph functions use Results internally and pack successful
outputs at
their `tvm_ffi.return` boundary. Graph rejection returns a successful GuardMatch
payload; the guard wrapper calls the graph through its packed ABI.
An in-place ATen call operates on the original object, and views keep their
storage aliases. ATen calls use the schema to
order positional and keyword arguments, fill defaults, select a scalar overload
when FX recorded a Python scalar in a tensor overload, and name the existing
`trident.aten.<operator>[.<overload>]` registry wrapper. Dtype arguments pass
through `trident.runtime.tvm_ffi_to_torch_type` to obtain the integer expected by
ATen's boxed schema. Multiple ATen results are retrieved from an FFI Array.

SymInt, SymFloat and SymBool values use TVM FFI Int, Float and Bool throughout
Python construction, including constants, symbol bindings and graph-node results.
Shared Python arithmetic builders in `ir.py` own type unification and emit native
`arith` operations. They extract temporary native `i64`, `f64` and `i1` SSA values
with `tvm_ffi.get`, widen operands and immediately wrap results with `tvm_ffi.to`.
Bool-to-Int uses zero extension, Bool-to-Float uses unsigned conversion and
Int-to-Float uses signed conversion. Metadata builders likewise wrap
native tensor metadata before returning it; dtype and device fields are exposed
as TVM FFI integer values. Native scalars are confined to local computation and
explicit SCF, CF and GPU interfaces.
Input tensor dimensions bind symbols to the semantic size builder; explicit
scalar nodes bind their expressions to imported semantic SSA values. Composite
shape expressions are evaluated from those bindings. Runtime expressions never
use example hints; unbound symbols produce an explicit import error.
`torch.cond` imports its branches into `scf.if`, extracting only its predicate
and retaining monadic Result values.

`ir.py` shares scalar arithmetic builders, constants, metadata and Result
construction with the guard frontend. Numeric constants use
`tvm_ffi.constant.bool/int/float` directly. `make_result_types` wraps CABI
payload types and supplies the None default for empty sequences. `pure` infers
CABI Result types from semantic payloads unless the caller declares a boundary
type, and widens semantic values at Any/Union boundaries. Neither Result nor
container construction implicitly accepts native scalar types or values.
Bind arguments retain their CABI types; consumers extract native scalars only
within a local builder or at an interface requiring them. Empty successful sequences produce one explicit None payload. Shared `checked_call` and
`container_get` builders construct registry calls and typed container extraction
without depending on the graph importer. The shared `BINARY_ARITHMETIC` operation
mappings also define the supported FX scalar operations; the importer only normalizes symbolic min/max
aliases.
Each asserting FFI call checks lookup success, a non-null handle and call success;
a failed assertion terminates execution. Generic failed Results still propagate
a nonzero packed ABI status unless handled explicitly at a graph-call boundary. A missing cached global establishes a RuntimeError
on the calling thread; resolved globals retain their cached fast path. Integer
division uses floor semantics; integer remainder follows
`a - floor(a / b) * b`. Supported arithmetic follows the runtime's 64-bit scalar
ABI. Scalar add, subtract, multiply, min/max, comparisons, bitwise operations,
shifts, true/floor division, remainder, negation, logical not and nonnegative
constant integer powers are built in Python using native arithmetic and boxed
results. `sum` composes scalar additions. Bool promotes to Int
for arithmetic, and either Float operand promotes numeric arithmetic/comparisons
to Float; true division always returns Float. Bitwise Bool/Bool retains Bool;
other bitwise operands, shifts, floor division and remainder require integral
inputs. Comparisons return Bool. The Python `logical_not` builder accepts Bool,
Int and Float and returns Bool with Python truth semantics: zero, including negative floating
zero, is false; nonzero values, infinities and NaN are true. Container `not`
extracts a length before applying the same Python builder.
Powers return Float for Float bases and Int otherwise, including exponent zero. Python builds integer powers by
squaring; floating powers retain sequential multiplication order. Floating
comparisons use ordered predicates except unordered not-equal; floating min/max
retain `arith.minimumf/maximumf` NaN behavior. Integer arithmetic remains fixed
width, with native overflow/division/shift constraints; this change does not
introduce arbitrary-precision integers or recoverable arithmetic exceptions.
Arithmetic failures are not specialization rejection Results. Unsupported scalar
expressions or higher-order operators fail explicitly.

TVM FFI types have concrete Python bindings exported by
`trident.core.dialects.tvm_ffi`, with TypeID registration for parsed types and
operation value types. Inspect them through `isinstance`, parameter accessors,
MLIR Type equality, and native integer width/signedness rather than text.
Textual parsing uses `ir.Type.parse` directly.

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

## Reusable lowering entry points

Pattern-based pass headers expose population interfaces for composing custom
lowering flows.
`populateDecomposeTVMFFIConversionPatterns` and
`populateLowerTVMFFIMonadConversionPatterns` each register default type rules,
patterns and target legality in one call using a caller-owned
`mlir::TypeConverter`. Add any overriding type rules after population. Keep the converter at a stable address and
alive through conversion; patterns and legality callbacks refer to it. After
successful operation conversion, recursively replace nested type attributes with
`mlir::AttrTypeReplacer`, as the standalone passes do.

`populateLowerTensorMetadataPatterns` adds ordinary rewrite patterns.
`populateLowerToPackConversionPatterns` and
`populateTVMFFIToFuncConversionPatterns` add patterns and target legality.
DLPack-to-LLVM, TVMFFI-to-GPU and TVMFFI-to-LLVM retain their existing population
interfaces. Callers load the dependent dialects declared by the corresponding
passes before applying these entry points.

Pattern-based passes use these same population interfaces internally. Deferred
cleanup and the lowering pipeline do not use patterns and expose only their pass
interfaces.

## Guards and containers

Guards are parsed into `GuardCode`, `Guard` and collection objects under
`python/trident/guards/`. `InputTableBuilder` maps function arguments and nested
list/tuple paths to semantic values. Lists use `!tvm_ffi.array<T>` with a common
semantic element type, or `!tvm_ffi.any` for empty or heterogeneous lists. Tuples
retain one type per element. `array_type` infers a common semantic element type;
`TupleType.get` preserves each semantic element type independently. Nested
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
builds fresh complete Results for a path; `extract_many` reconstructs leaf
inputs in exported order. Extraction is lazy in its consumer block. Guards use
these table methods and shared IR builders directly, without a graph importer.
Tuple length guards retain checked runtime `ffi.ArraySize` calls: wrapper inputs
must validate arity before trusting their static tuple schema.
A single input-node implementation handles leaves and containers; each fresh
table holds explicit root or consumer-local payload bindings without caching
extraction Results. Scalar guards return TVM FFI values from local arithmetic and comparisons; tensor metadata uses existing semantic
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
attributes and assembles operands. Ordered launch construction clones functional
outputs, inserts the binary and emits the launch, returning complete Results
for its output tensors.
Each specialization has a unique binary symbol. Operands retain semantic FFI
values and carry constant or variable specialization attributes in source
parameter order. Variable attributes record native ABI type and divisibility;
constant attributes include scalar, string and nested tuple values. Binder hints
are used only to select the compiled kernel, never to substitute runtime shapes.
Functional Triton wrapper outputs explicitly clone `tensors_to_clone` before
launching; mutation wrappers use the original operands.

`tvm_ffi.kernel_launch` returns `Result<None>`. `lower-kernel-launch`
expands specialization attributes into checks and Monad pure/fail Results. Its
success branch contains `tvm_ffi.native_kernel_launch` with native operands and
no specialization attributes. `ConvertTVMFFIToGPU` lowers only that unconditional
operation to `gpu.launch_func` with the current TVM FFI CUDA stream.
That branch returns successful None; rejection returns failure without setting a runtime error or exiting the function.
Constant operands are checked and omitted from the native argument list.
Python binds the launch result, skipping later computation on rejection.
Success means the launch path was taken, not asynchronous execution completed.

## Lowering and ownership

The lowering pipeline is:

1. Run `lower-kernel-launch` to remove specialization attributes and construct
   guarded native launches with Monad success/rejection Results. Then run
   `convert-tvm-ffi-to-gpu` for stream lookup and GPU launches.
2. Run `lower-tvm-ffi-monad`. Every `Result<T>` becomes a semantic
   `tuple<bool, any>`. Pure values become `(true, payload)` and failures become
   `(false, None)`. Binds become tuple extraction and `scf.if`; their typed
   payloads are restored only in successful branches. Function signatures,
   calls and control-flow arguments are converted together.
3. Run the MLIR canonicalizer after monad lowering. Its TVM FFI tuple pattern
   forwards elements through a single-use `tuple.construct`/`tuple.unconstruct`
   pair, so transient Result tuples do not reach runtime Array construction.
   Tuples that remain observable still retain their normal container lowering.
   Then run `decompose-tvm-ffi`. Preserve local `tvm_ffi.call` operations and
   convert semantic object types to their Base representations throughout
   signatures, regions and type-bearing attributes, including Array/Tuple to
   `!tvm_ffi.c.array` and Any/Union/Str to `!tvm_ffi.c.any`. Container operations become
   checked runtime Array calls. Materialize semantic export retains and register
   local releases in defer regions. Run `lower-tvm-ffi-defer` immediately
   afterward to clone their bodies
   at dominance-scope exits before inlining and SCF-to-CFG conversion.
4. Run `lower-tensor-metadata` to expand tensor metadata operations
   through borrowed object/DLTensor views into DLPack operations. Then run
   `lower-to-pack` followed by
   `convert-tvm-ffi-to-func`. The first pass replaces every typed `tvm_ffi.func`
   body with a `tvm_ffi.pack.func` entry of the same name and visibility. It checks argument count before reading
   slots and checks tags before entering the original CFG. Rejection returns a
   successful GuardMatch. Original successful returns retain their declared
   payload checks, while the packed signature returns CAny to admit GuardMatch.
   The second pass converts packed entries to `__tvm_ffi_<name>` functions with
   output slots and an i32 status, preserving the original visibility. External
   declarations must specify visibility accepted by `func.func`, which rejects
   public declarations. Output slots are written only on success;
   there is no separate typed implementation or wrapper call.
5. Convert local typed and packed calls in TVMFFIToFunc to direct calls of
   the generated packed ABI functions. Lift the packed preflight CFG to SCF
   before the default inliner and symbol DCE: multi-block callees otherwise
   cannot inline into single-block SCF branches. DLPack metadata operations
   support inlining. Run CSE and canonicalization before SCF-to-CF to share
   identical predicates and propagate conditions established by enclosing
   branches. Scalar boxing/extraction and matching widen/narrow casts fold,
   including native constants extracted from TVM FFI scalar constants, so they
   do not hide constant or repeated conditions from SCF simplification.
   Calls rejected by the inliner retain their packed ABI. Then run GPU-to-LLVM,
   TVMFFI-to-LLVM and the remaining LLVM conversions, canonicalization and cast
   reconciliation. Ownership cleanup is materialized before either inliner.

`tvm_ffi.monad.call @callee(...)` returns `Result<T>`, where T exactly matches
its callee's return type. `lower-tvm-ffi-monad` emits `tvm_ffi.call @callee(...)`
with separate `!tvm_ffi.bool` and `!tvm_ffi.any` results, then explicitly packs
them into the semantic Bool/Any tuple used for lowered Results. The tuple
borrows the call payload and retains it through runtime container construction.
Decomposition releases both the call's independent reference and its local tuple
reference at dominance-scope exit, retaining any exported tuple first.
Result attributes move to the call's payload result, with an empty entry for success.
Local callee resolution and argument matching are deferred to lowering for
unlowered callees. This local-symbol call is distinct from registry
`tvm_ffi.FunctionCall`.

`decompose-tvm-ffi` preserves the call's two results while converting container
operands and explicitly constructed tuples to CArray. `LowerToPack` converts
typed function bodies to packed entries. `TVMFFIToFunc` lowers packed definitions
to `__tvm_ffi_<name>` ABI functions and converts local calls to direct calls of
those entries. The pipeline inlines these calls before lowering SCF regions.

`TVMFFIToFunc` requires decomposed container operands and expands the call through
`__tvm_ffi_<callee>` using packed argument and output slots. It emits a direct
`func.call` to the packed entry supplied by `TVMFFIToFunc`; it does not look up,
validate or create entry declarations. Zero status produces separate
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
Private graph functions and guard wrappers use local packed calls. Function
payloads are boxed into Any during monad lowering.

`TridentFxImporter.visit` iterates over the original FX nodes. Node handlers
construct only their current node and return its ordinary value structure, whose
leaves may include complete Results. The visitor flattens this structure, binds
its Result leaves, restores payloads in place, and continues the loop inside the
successful body. Unit Result leaves with a None payload restore to Python `None`,
consuming their corresponding block argument. The visitor does not redispatch on
the node target after binding. Local `BindScope` contexts keep nested insertion
points open; the visitor explicitly yields complete Results from the innermost
bind outward. Final outputs use `pure`, including a None payload for no-data
returns.

Ordinary scalar, schema and metadata helpers return ordinary values. Independent
fallible helpers construct arguments, checked calls, ATen conversions and result
extraction, inputs, or guard expressions and return complete Results. One
argument builder automatically converts Python `torch.dtype` arguments to ATen's
integer enum, including elements of nested lists and tuples. Existing SSA values
keep their types; ordinary dtype constants and guard expressions retain the TVM
FFI dtype representation.
Ordered argument, input, guard-operand and result extraction builders carry
accumulated payloads into each successful bind and wrap them once at the end.
They use local `BindScope` contexts for ordered binds and recurse only into
nested list/tuple elements, preserving element order and failure short circuiting.
Empty argument and input builders return one None Result to preserve success
tracking. Consumers use the original list length to omit this placeholder from
actual call arguments; explicit Python None arguments retain their positions.
Argument construction does not invoke an operator or traverse later FX nodes. Kernel clone, grid and
launch construction likewise preserves its original order.

The shared `trident.ir.Scope` abstraction manages one block's insertion point
and exposes complete operation results. `BindScope` constructs one `MonadBindOp`;
its `arguments` expose borrowed successful payloads in input order, including
None payloads. Explicit `scope.yield_(results)` terminates only its own block;
`scope.results` exposes complete Results that may leave the region.
FX traversal uses a local `ScopeStack` to keep subsequent nodes inside successful
bind bodies without recursive Python calls. Other iterative builders use a stack
when they accumulate a dynamic chain of binds; single binds use `BindScope`
directly. `stack.bind(result_types, inputs)` returns successful arguments, and
`stack.yield_(values)` drains scopes from innermost outward, forwards complete
results and immediately restores insertion points. An empty stack forwards values
unchanged. Independent child stacks leave parent scopes open. Context exits
clear stored scopes and restore insertion points without generating terminators,
including on exceptions. ScopeStack uses the standard-library ExitStack for
restoration. Simple SCF branches and function bodies use insertion points and
explicit `scf.yield`, `func.return` or `tvm_ffi.return` directly.
Decomposition manages the lowered Result containers automatically.
The underlying `MonadBindOp.arguments` property exposes the successful payloads,
while `outputs` exposes complete Results.
The property is also available on parsed and traversed OpViews. MLIR verification
checks the successful terminator and yielded Result types. Each `BindScope`
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
There are no continuation callbacks, Effects objects, generator resume protocols
or task queues. Linear fallible chains do not add Python recursive frames.
Nested containers, expressions and control-flow branches remain recursive.
`import_program` and direct `visit` calls retain their value and symbol mappings;
use a fresh importer for each program.

Python builds ordinary operations and exits without ownership attributes.
Function entries and SCF branches use insertion points directly. Cond
branches use independent importers sharing the parent module, and return Results for the
parent visitor to bind. Input tables construct fresh complete Results for each
extraction at the current insertion point, without caching Results or bind payloads.
Explicit root and consumer-local payload bindings remain available within their
lexical scopes. MLIR verification checks SSA visibility; frontend builders do not
perform separate block-ancestry checks. Semantic reference lifetimes extend to
the defining IR block's dominance-scope exit, independently of Python context-manager exits.
Guards and backend input flattening use fresh tables. Construction exceptions propagate; discard
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
guarded branch, binds flattened inputs and forwards the packed graph result. Each `tvm_ffi.func` declares exactly one C ABI payload type;
`tvm_ffi.return` carries a boxed success boolean and Any payload. Generic no-data
functions can declare `!tvm_ffi.none`; generated graph and wrapper functions
declare Any so they can also return GuardMatch. Private graph helpers bind
internal Results, pack successful payloads and use `tvm_ffi.return`. Packed
output slots are read and written only on success. Multiple
values are packed into a semantic tuple payload before returning. Guard
rejection and Triton specialization failure remain successful `GuardMatch`
values, requesting another specialization; runtime FFI failures return nonzero
status without inspecting output payloads.

Lowered Results are runtime Array tuples containing a boxed boolean and payload.
Failure tuples contain None. Both the Python `pure` helper and low-level
`MonadPureOp` borrow their inputs: tuple construction retains each occurrence,
including repeated payloads. No transfer mode or ownership attribute is needed.

`decompose-tvm-ffi` uses conversion legality rather than a parent-operation
whitelist. Synchronous nested regions may borrow captured semantic references;
local producers register cleanup in their own region. Region boundaries carrying
semantic values require dedicated ownership conversion patterns. Functions,
explicit defer bodies, `scf.if`, `scf.for` and `scf.while` are supported;
unsupported semantic boundaries fail legalization. Cleanup region structure is
validated by `lower-tvm-ffi-defer`. This policy assumes synchronous execution;
asynchronous reference lifetimes are outside the automatic ownership contract.

Each supported SCF operation has its own conversion pattern. Parent patterns
convert their operands, results and region signatures, move existing regions and
register result cleanup. The for and while patterns retain initial values and
register per-execution entry argument cleanup. This also preserves results when
a for loop executes zero times. Independent yield and condition conversion
patterns retain each outgoing semantic slot using their own remapped adaptors.
Yield ownership applies to if, for and while boundaries. The condition predicate
is not retained; both condition outcomes share the same outgoing payload
reference. Repeated semantic slots retain separate references.

Parent patterns inspect original semantic types before converting signatures
through the conversion rewriter. Child operations, including terminators, are
left to their own conversion patterns.
There is no generic region-boundary conversion or module-wide ownership plan.
Function arguments, implicit captures and internal CFG block arguments borrow;
loop entry arguments receive owned references and release them on every region
execution, after outgoing references have been retained.

All decomposition patterns derive directly from `mlir::OpConversionPattern`.
Borrowing patterns use the rewriter's replacement methods; producers explicitly
register local releases for original Opaque result types in a `tvm_ffi.defer`,
in result order, before replacing the operation. Representation-only
patterns use `ConvertGenericOp<Op, OwnsResults>`, with ownership disabled by
default and explicitly enabled for producers at registration. The replacement
path is selected at compile time without runtime producer classification.
Conversion legality depends on types; defer lowering validates cleanup region
structure. Casts, function parameters, internal CFG parameters and captures borrow;
forwarded region entry parameters own their received references. Tensor storage
aliases do not imply borrowed object references.

Function returns retain every Opaque operand in operand order, including repeated
values, borrowed parameters and values widened from Base or POD types. Region
exports also retain each forwarded Opaque slot, including direct producer results.
Producer references always receive local cleanup; there is no implicit reference
transfer optimization. Original Base slots remain manually managed even when
their representation matches an Opaque slot.

The decomposition implementation separates representation mapping, container
runtime calls, ownership completion and conversion configuration. Array and Tuple
construction share one pattern; container patterns explicitly release their
temporary Base function handles after each runtime call. Producer patterns
register semantic cleanup through the driver-provided rewriter in the same
conversion transaction. Value replacements register cleanup at the current
insertion point; operation replacements place it
after the replacement operation. Dedicated function return patterns retain exported
operands without applying producer ownership rules. Dedicated for and while
patterns insert initial-value retains and entry cleanups. The if, for and while
patterns directly register result cleanup. Independent yield and condition
patterns retain transferred payloads and replace their own terminators using
adaptor values. Patterns retain original Opaque slots directly; shared helpers register
deferred releases in slot order. Patterns do not analyze region control flow.
The pass converts operations, then recursively converts type-bearing attributes.

Patterns use original types to classify ownership. Retains and cleanup
use remapped operands and replacement values, including child exit retains
whose values come from the terminator pattern's adaptor. Any/Union become CAny,
so type legality forces their ownership patterns to run once without tracking
completed replacements. The pass uses the default dialect conversion configuration:
rollback remains enabled, preserving original operand types until commit, and
folding is attempted before patterns. Region conversion moves existing child
operations rather than cloning them.

Patterns use default benefits unless they compete for the same operation.
Monad lowering gives its attribute-preserving `scf.yield` conversion priority
over the upstream structural conversion. LLVM constant PDLL patterns use default
benefits and retain `PDLConversionConfig` so `convertType` uses the LLVM type
converter.

`lower-tvm-ffi-defer` clones cleanup bodies at registration dominance-scope
exits, after all export retains. A registration remains active through blocks
dominated by its registering block. An edge leaving that dominance scope runs
the cleanup; an edge back to the registering block also cleans up the previous
iteration before registering again. Region returns and yields clean up all
remaining active registrations. Relative execution order between defers at the
same exit is unspecified; operations requiring an order must share one body.
Operation order and SSA data flow within each body are preserved. Edge-specific
cleanup blocks preserve branch operands, including distinct edges to the same
successor. Runtime-tagged reference
checks handle CAny payloads, including scalar-only values. Temporary Base function
handles created by decomposition retain their immediate, explicit releases.
Original Base results remain manually managed, including Base-typed container
extractions. A manually managed reference exposed through an Opaque return or
yield still receives an independent export reference.

Automatic management supports multi-block function CFGs and the three supported
SCF operations. CFG block arguments remain borrowed: forwarding a reference across a
branch does not extend the owner's dominance scope or transfer ownership.
Code using such a value after its owner scope ends must already hold an independent
reference. Internal CFG forwarding does not infer ownership; region entry
forwarding instead follows the operation-specific rules above.
Explicit defer lowering supports loops as described below.

The dedicated `lower-tvm-ffi-defer` pass runs immediately after decomposition.
Each defer body has a single block without arguments, captures its inputs, and
ends in `tvm_ffi.defer_yield`. Nested regions and additional terminators in cleanup
bodies are rejected. Cleanup does not unwind aborts or exceptions.
Explicit defer lowering supports any SSACFG region with block terminators,
independent of the enclosing operation or exit kind. MLIR dominance analysis
must establish SSA semantics; graph regions and unknown single-block regions are
rejected. CFG branches must implement `BranchOpInterface`. A terminator without
CFG successors ends this execution of the region, so all active cleanups run
before it. This includes `scf.condition` on either outcome and transfers to
another region. Each region execution has independent registrations; a zero-trip
for-loop executes no body cleanup. No operation-specific adapters are required.
This pass does not retain exported operands or infer loop-carried ownership.
Callers using explicit defer must independently preserve forwarded references.
Unsupported region structure and CFG branches within active scopes are diagnosed
before any cleanup is cloned or CFG edges are changed. Unreachable registrations
are removed without generating cleanup. Empty bodies and edges requiring no
cleanup may produce forwarding blocks, which later canonicalization can remove.
Automatic cleanup registration may emit an argument-free block containing only
`tvm_ffi.defer_yield` when no values need releasing; this is a valid empty defer.
Decomposition remains a once-only lowering boundary; defer lowering is a no-op
when no defers remain. There is no early-release optimization or complete borrow
checker; callers must preserve reference validity.

Generated private graph Results use failure for specialization rejection. At the
graph function exit, the importer checks all Result success bits,
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

`tvm_ffi.isa` tests Any/CAny against one concrete runtime type index without
trapping or retaining the value. Semantic Array/Tuple, Exception, Function and
Tensor targets decompose to their indexed Base types. Container element types
are not checked; Any, CAny, Str and Union targets are rejected. Ordinary local
`tvm_ffi.call` operands must match the typed callee parameters. Dynamic
`tvm_ffi.pack.call` forwards the complete argument view; the converted packed
entry checks count before loading slots and then checks tags, returning a
successful GuardMatch on mismatch.


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

## Dynamic packed entries

`!tvm_ffi.pack.arguments` borrows the packed ABI argument pointer and i32 count.
It is outside the tagged C ABI and cannot be a Result/container payload or a
function return. C and Python bindings expose construction, type checks and
TypeID registration. `pack.func` requires exactly this one input and one CABI
payload result; it always exports `__tvm_ffi_<name>` without an ABI marker.
The converted native `func.func` also needs no ABI marker;
packed calls resolve the generated ABI symbol directly.
`pack.size` sign-extends the count to native i64. `pack.get` takes native i64,
asserts `0 <= index < size` before loading, and returns borrowed Any.

`pack.call` forwards the original pointer and count to a local typed or packed
entry without allocating or expanding an argument array. Entries converted from
typed functions reject wrong counts before reading slots and reject wrong tags
with GuardMatch. Both conversion passes run after decomposition; typed parameters
must be indexed Base types or CAny. Native parameters cannot be packed.
Calls return Bool/Any; failure preserves the runtime error and returns None
without reading the output slot. Successful call payloads own an independent
reference. Pack views and reads own no references; existing semantic consumers
and exports retain borrowed values when needed. Ordinary `tvm_ffi.call` still
requires matching typed operands and has no Any-to-arbitrary-type exception.

Dynamic arity is supported at the MLIR packed entry. Python `*args`/`**kwargs`
normalization and guard input mapping require separate verification; the ABI
alone does not establish support for those frontend signatures.
