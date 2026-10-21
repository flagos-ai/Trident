<!--
Part of the Trident project, under the MIT License.
SPDX-License-Identifier: MIT
-->

# Core Lit tests

Test directories mirror their owning implementation directories under `core/lib`:

- `Conversion/<Component>/` covers local conversion behavior. When one component
  implements multiple passes, keep their tests together and distinguish the
  passes with descriptive filename prefixes rather than inventing source
  directories.
- `Conversion/Pipeline/` covers representative end-to-end lowering through
  `--trident-lowering-pipeline`, and explicit pass compositions when their
  integration contract concerns an intermediate dialect such as `gpu`.
- `Dialect/<Dialect>/IR/` covers parsing, printing, verification and operation
  semantics, including effects checked through canonicalization. Invalid IR uses
  `-verify-diagnostics`.
- `Dialect/<Dialect>/Transforms/` covers dialect transformation passes.

`Dialect/TVMFFI/Transforms/` covers `lower-tvm-ffi-monad`, which converts
Results to Bool/Any tuples and binds to SCF branches. `monad.call` becomes a
two-result `call` followed by borrowing tuple construction. Decomposition
registers the call payload and tuple releases in defer regions.
`lower-tvm-ffi-defer` moves those bodies before their enclosing block terminators before
inlining or SCF-to-CFG conversion; explicit test pipelines must include it after
decomposition.
`decompose-tvm-ffi` preserves local calls and erases semantic containers. `TVMFFIToLLVM` expands local calls through the packed ABI.
`lower-monad-call.mlir` checks the monadic interface boundary;
`decompose-call.mlir` checks call preservation, container type conversion and
attributes. `Conversion/TVMFFIToLLVM/call.mlir` checks packed call status,
output-slot gating and the absence of runtime Array construction; `call-invalid.mlir` checks
missing decomposition.
`Conversion/Pipeline/checked-continuation.mlir` covers integration with checked
calls and the full lowering pipeline. No Result types reach LLVM conversion.

Keep each file focused on one operation family or lowering concern. Use
descriptive filenames, `CHECK-LABEL` for each function, and named FileCheck
captures when a value is reused. Bind function arguments and reuse those
captures when checking data flow instead of relying on printed names such as
`%arg0`. Spell FileCheck directives with one space after the colon and use
`[a-zA-Z0-9_]+` for ordinary SSA captures. Do not use anonymous wildcard
matches in place of a capture. Shared auxiliary inputs belong in an `Inputs/`
directory and are excluded by the root Lit configuration.

Use the smallest pass sequence for local conversion tests. Parsing and
printing belong in `Dialect/<Dialect>/IR/`, rather than an extra no-pass `RUN`
in a conversion test. Metadata and GPU tests composing multiple conversions belong
in `Pipeline`; the GPU pass-local test checks semantic operands before ABI
lowering, while its pipeline counterpart checks native launch operands.

`Pipeline/aten-dispatch.mlir` owns the representative whole-pipeline registry
call, cached-handle ownership and packed wrapper checks. Numerical overloads,
memory formats and view aliases are checked through the public Python JIT;
they do not each need a copy of the same ABI fixture. Dimension arrays,
semantic copy, tensor literals and control flow retain their own integration
regressions. Ownership edge transfers and ABI zero initialization remain
explicit pass-local contracts. Result conversion and checked returns have focused conversion regressions.
`bind-arities.mlir` covers zero-input and zero-output binds through parsing,
decomposition and LLVM lowering, including conditional side effects. Result
verifier tests retain argument and yield matching checks for these cases.
Python ownership tests cover attribute-free resource construction, borrowed
pure payloads, repeated exports, successful and failed Results, independent Bind
arities and lexical cache reuse. `automatic-ownership.mlir` checks borrowed
parameters, cast aliases, unused owned results, mixed extraction, reverse release
order, all export retains before local releases, and manual Base references.
`automatic-ownership-invalid.mlir` checks unsupported multi-block ownership.
`decompose-pattern-ownership.mlir` checks Any/Union-to-CAny rewrites,
constant-branch ownership before folding, nested Opaque-valued joins, mixed
Opaque/Base branch results, borrowing casts without local cleanup, manual Base
registry results and zero-result extraction. Together with mixed extraction
coverage, these exercise value, operation and newly constructed replacements. Opaque exports retain their
values regardless of their source, including casts from Base and scalar values.
`lower-defer.mlir` checks cleanup ordering, branch isolation, empty bodies and
repeat lowering. Defer verifier and lowering diagnostics have focused negative
tests, including multi-block regions, unsupported enclosing operations and
unsupported exits. Both decomposition and defer lowering require single-block
lexical cleanup scopes.
Container, monad-call and pipeline regressions cover automatic management through
representation conversion and LLVM lowering. CUDA e2e allocation checks enforce
cached-execution leak freedom for PyTorch-managed allocations.

Python tests never load these files as executable fixtures. Lit regressions
can be minimized or renamed independently of Python test setup.

TVM FFI functions declare exactly one C ABI payload type and return a boxed
success flag and Any payload through `tvm_ffi.return`. Return coverage includes None and tuple payloads, output-slot
status gating. Result reference operations access payloads only in successful branches.

`decompose-c-any.mlir` covers erased Union/Str downcasts and boundaries,
recursive Result payload conversion and manually managed CAny values. Finite
Union/Str runtime checks are deferred; concrete indexed Base downcasts retain
their existing coverage. CAny is a Base tagged representation without a single
type index.
