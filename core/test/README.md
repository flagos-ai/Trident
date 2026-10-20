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
two-result `call` followed by explicit tuple construction and payload release.
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
Python ownership tests cover explicit operation cleanup, explicit exports and
independent Bind input/output counts. `explicit-cleanup.mlir` checks UnitAttr
markers, mixed results and transfers;
`decompose-cleanup.mlir` checks explicit cleanup propagation to independent call
references without duplicating temporary-handle releases.
`decompose-explicit-cleanup.mlir` checks marked and unmarked construction,
extraction and calls, including empty tuples and scalar results.
`deferred-container-ownership.mlir` checks borrowed CArray exports, independent
nested extraction, repeated borrowed pure payload retains and idempotence through
decomposition and cleanup without a second bind expansion.
`materialize-cleanup.mlir` checks block-exit release order, nested scopes, separate
Result success conditions and idempotence.
`result-reference-adjustments.mlir` checks direct unpack and success branches
for Any/Union and Function payloads, failed cleanup and LLVM lowering.
`retain-exports.mlir` covers four explicitly marked exit kinds, repeated and borrowed operands, containers, failed Results, scalar-only
and empty exits, retain-before-release order, consumed export markers and idempotence.
`decompose-export-markers.mlir` checks pending markers on converted returns and
yields, unmarked generated failure exits, and cleanup idempotence after bind
expansion.
`two-stage-cleanup.mlir` checks that all exports are retained before any local
release, single-block `scf.execute_region` support and unchanged scalar markers
in skipped multi-block regions. `skip-multiblock-cleanup.mlir` checks that skipped
multi-block regions keep their direct operations unchanged while visited nested
single-block regions are processed independently.
Pipeline tests check that exports acquire runtime
reference operations. Lit also checks lowering-local handle cleanup. Kernel tests check Result<None> success/rejection without an early
function exit. CUDA e2e allocation checks enforce cached-execution leak freedom.

Python tests never load these files as executable fixtures. Lit regressions
can be minimized or renamed independently of Python test setup.

TVM FFI functions declare exactly one C ABI payload type and return a boxed
success flag and Any payload through `tvm_ffi.return`. Return coverage includes None and tuple payloads, output-slot
status gating. Result reference operations access payloads only in successful branches.
