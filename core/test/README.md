<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Core Lit tests

Run `cmake --build build/core --target check-trident-core`, or the equivalent
`core-build` target for a standalone configuration.

Directories mirror the implementation layer:

| Directory | Contract |
| --- | --- |
| `Dialect/<Dialect>/IR/` | Parsing, printing, verification and operation semantics |
| `Dialect/<Dialect>/Transforms/` | Dialect transformation passes |
| `Conversion/<Component>/` | Local conversion behavior |
| `Conversion/Pipeline/` | Representative cross-pass integration |

## Coverage ownership

Keep one file per operation family or conversion concern. Related cases share
RUN lines and use `--split-input-file` where modules or diagnostics need isolation.
A new operation or regression normally extends an existing family instead of
adding a file for each type, arity or operator name.

| Family | Coverage boundary |
| --- | --- |
| `DLPack/IR/ops`, `TVMFFI/IR/types*` | Public type spellings and representative type parameter constraints |
| `TVMFFI/IR/ops*` | Scalar/registry operations, containers, reference types, cleanup regions and kernel syntax/verifiers |
| `TVMFFI/IR/casts*` | Widening, membership, subsets, identity parsing/folding and incompatible casts |
| `TVMFFI/IR/result*` | Local calls, variadic Results, bind/yield, unpack inference and packed return shape |
| `Transforms/lower-monad` | Result construction/unpack, continuations, independent input/output arities, local call attributes and CFG signature conversion |
| `Transforms/decompose-representations` | Opaque/Base mapping, recursive Result/attribute conversion, borrowing casts, local calls and string elements |
| `Transforms/decompose-containers` | Checked container calls, nested types/attributes and captured values inside nested continuations |
| `Transforms/automatic-ownership*` | Producer cleanup, borrowed aliases, repeated exports, mixed Opaque/Base results, region transfers, loop-carried references, direct producer exports, CFG type conversion and synchronous captures and rejected legalization |
| `Transforms/lower-defer*` | Generic SSACFG dominance-scope cleanup, CFG edges/loops, generic region exits, SSA cloning, empty bodies, repeat lowering and rejected region structure |
| `DLPackToLLVM/metadata`, `Transforms/lower-tensor-metadata` | Tensor metadata expansion to DLPack and LLVM field mapping |
| `Transforms/lower-to-pack` | Typed-body conversion, count/tag guards, CFG forwarding, success-only payload checks and visibility |
| `TVMFFIToFunc/abi-and-cfg`, `return-invalid` | Packed entry ABI, output writes, return status, declarations and duplicate-symbol verification |
| `TVMFFIToGPU/kernel-launch`, `invalid` | Launch arguments/stream, constant/variable specializations, Result success/rejection and unsupported native operands |
| `TVMFFIToLLVM/call`, `invalid`, `pack` | Packed argument/output slots, success-only reads, zero/repeated calls, borrowed pointer/count forwarding, checked dynamic indexing and undecomposed input rejection |
| `TVMFFIToLLVM/casts*` | Observable concrete tag assertions, erased finite constraints, string casts and Base identity conversion |
| `TVMFFIToLLVM/value-ops` | Representative constant layouts, scalar boxing/extraction, kernel-argument truncation, object pointers and tensor literals |
| `TVMFFIToLLVM/runtime-ops` | Cached registry lookup/error/call checks, conditional reference updates and clone/copy operand data flow |

Public type mnemonics retain individual spelling checks because they have separate
parser registrations. Generic constraints use representative categories rather
than every scalar width or object type. Result payload rejection retains native
scalar, native Object and nested Result examples. Unpack inference retains Any,
CAny, a scalar and a native function handle; cast compatibility owns membership
and subset variations. Identity folds use scalar, container, function and union
representatives. Scalar constants reuse boxing coverage; DType packing and None
initialization keep their own checks.

Automatic ownership checks producer defer registration (`DEFER`) and final cleanup
(`CHECK`); region transfers check the resulting retains and releases directly. Original Base values stay manually managed. Exports retain independent
references before local releases, including repeated exports and branch
results. Region tests cover loop initialization, per-execution entry cleanup,
results and condition forwarding in if/for/while, with representative full LLVM
lowering. Synchronous execute_region captures and local cleanup are accepted;
negative cases reject unsupported semantic execute_region and index_switch
boundaries. Base-only regions remain accepted. Internal CFG
block arguments borrow; branch forwarding does not extend owner
lifetimes. CFG cleanup runs on dominance-scope exits and registration-block
re-entry, without prescribing relative order between defers at the same exit.
Empty and unreachable registrations are covered together; upstream CFG
canonicalization does not need its own cleanup fixture.
Monad lowering precedes decomposition; defer lowering immediately follows it,
before inlining or SCF-to-CFG conversion. Semantic tuple inputs rely on this
policy without manual retains.

Pipeline cases are selected by interaction:

| Fixture | Integration contract |
| --- | --- |
| `aten-dispatch` | Semantic call ownership, cached dispatch and packed entry |
| `c-any-checks` | Erased finite constraints at packed boundaries |
| `checked-continuation` | Nested status-gated calls and payload-independent mutation |
| `containers` | Array metadata, tuple construction/extraction and packed tuple results |
| `kernel-launch` | Specialization, tensor/native arguments and runtime stream |
| `local-call` | Monadic local calls through converted packed entries |
| `tensors` | Dynamic metadata and splat/nonsplat tensor literals through LLVM |

Local passes own scalar boxing, None returns, copy operand order and string casts.
The pipeline does not repeat generic ATen dispatch for each operator name, or
upstream SCF selection and assertion lowering in isolation. Python execution tests
own numerical overloads, list-valued schema arguments, formats, aliases, allocation
behavior and output pytrees. Python tests never load Lit files as executable
fixtures.

## Writing tests

Use the smallest pass sequence that owns the behavior. Keep negative parsing and
verification cases with their operation family, using `--verify-diagnostics`.
A merged file must run every section through its intended pass sequence; use
separate files when pass requirements conflict.

Use `CHECK-LABEL` for each function and bind relevant SSA values, including
arguments, to verify data flow. Avoid incidental IDs and printed argument names.
Use `CHECK-NEXT` for intentional order, such as export retains preceding
local releases. Keep lookup status, non-null handles, call status, ABI zero
initialization and success-only output-slot reads explicit. Never replace SSA
captures with unconstrained wildcards. Use a single space after directive colons.
Shared auxiliary inputs belong in `Inputs/`, which Lit excludes from discovery.
