<!--
Part of the Trident project, under the MIT License.
SPDX-License-Identifier: MIT
-->

# Core Lit tests

The directory follows the LLVM convention of organizing
regression tests by the implementation layer under test:

- `Conversion/<Pass>/` covers one conversion pass and checks its pass-local
  rewrite. The directory name matches the corresponding `core/lib` pass.
- `Conversion/Pipeline/` covers representative end-to-end lowering through
  `--trident-lowering-pipeline`, and explicit pass compositions when their
  integration contract concerns an intermediate dialect such as `gpu`.
- `Dialect/<Dialect>/` covers parsing, printing, verification, and dialect
  transforms. Invalid IR uses `-verify-diagnostics`.

Keep each file focused on one operation family or lowering concern. Use
descriptive filenames, `CHECK-LABEL` for each function, and named FileCheck
captures when a value is reused. Bind function arguments and reuse those
captures when checking data flow instead of relying on printed names such as
`%arg0`. Spell FileCheck directives with one space after the colon and use
`[a-zA-Z0-9_]+` for ordinary SSA captures. Do not use anonymous wildcard
matches in place of a capture. Shared auxiliary inputs belong in an `Inputs/`
directory and are excluded by the root Lit configuration.

Use the smallest pass sequence for local conversion tests. Parsing and
printing belong in `Dialect`, rather than an extra no-pass `RUN` in a
conversion test. Metadata and GPU tests composing multiple conversions belong
in `Pipeline`; the GPU pass-local test checks semantic operands before ABI
lowering, while its pipeline counterpart checks native launch operands.

`Pipeline/aten-dispatch.mlir` owns the representative whole-pipeline registry
call, cached-handle ownership and packed wrapper checks. Numerical overloads,
memory formats and view aliases are checked through the public Python JIT;
they do not each need a copy of the same ABI fixture. Dimension arrays,
semantic copy, tensor literals and control flow retain their own integration
regressions. Ownership edge transfers and ABI zero initialization remain
explicit pass-local contracts.

Python tests never load these files as executable fixtures. Lit regressions
can be minimized or renamed independently of Python test setup.
