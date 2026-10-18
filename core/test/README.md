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

`Dialect/TVMFFI/Transforms/` covers `decompose-tvm-ffi`, which expands
binds throughout the module into unpack and SCF branches while retaining Result
types.
The `lower-tvm-ffi-monad-` tests in the same Transforms directory cover
`lower-tvm-ffi-monad`, which lowers
decomposed Results to payloads and success bits, including SCF/CFG structural
type conversions and return materializations.
Tests consuming binds run decomposition before conversion; the Result fixtures
cover payload/status expansion, failure and unsupported-op diagnostics.
`Conversion/Pipeline/checked-continuation.mlir` covers their integration with
checked calls and the full lowering pipeline.

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
explicit pass-local contracts. Monadic Result expansion, checked returns and
success-only return retains have focused conversion and ownership regressions.

Python tests never load these files as executable fixtures. Lit regressions
can be minimized or renamed independently of Python test setup.

TVM FFI functions declare Result signatures and return matching Results through
`tvm_ffi.return`. Return coverage includes unit and mixed Results, output-slot
status gating, packed payloads, and conditional ownership transfer.
