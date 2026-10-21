<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Python tests

Use the existing `unittest` runner. All test directories are ordinary packages;
there are no custom discovery hooks, generated test methods or compatibility
modules for the former flat layout.

| Directory | Contract | Execution prerequisites |
| --- | --- | --- |
| `unit` | Guard parsing, expression IR and importer diagnostics | Installed native extension; no GPU execution |
| `e2e` | Public JIT capture, guards, compilation and cached execution | CUDA |
| `support` | Fresh IR contexts and CUDA allocation checks | Helpers, never a discovered test suite |

## Running

From the repository root, with the editable build installed:

```bash
python -m unittest discover -s test -p 'test_*.py'
python -m unittest discover -s test/unit -t . -p 'test_*.py'
python -m unittest discover -s test/e2e -t . -p 'test_*.py'
python -m unittest test.e2e.test_specialization
cmake --build build/core --target check-trident-core
```

End-to-end tests require CUDA and report PyTorch errors directly if it is
unavailable. Selecting `unit` does not run CUDA tests.

## Writing tests

Organize operator execution tests in `e2e/test_<operator>.py`, keeping overloads
and parameter variants of the same operator together. Use the trailing underscore
for in-place operators, such as `test_add_.py` and `test_copy_.py`. Frontend
contracts spanning operators stay in `test_containers.py` and
`test_specialization.py`. Run one operator with
`python -m unittest test.e2e.test_clone`.

- Keep guard parsing and expression IR checks in `unit/test_guards.py`, and
  public behavior in `e2e`. Lit owns verifier and pass-local IR contracts. The
  same operator at different layers is not duplicate coverage when the
  assertions check different contracts.
- Use named `subTest` cases for equivalent flows. Keep the full meaningful
  operand-type, sign, layout and error matrices.
- Each test owns its JIT functions and IR context. Do not compile at module
  import or share mutable specialization state between tests.
- Public JIT tests check warmup and compiled calls, with specialization-count
  assertions for cache-reuse contracts. Use owned input tensors to avoid
  conservative `_base` guard misses; use preserve-format clone for an owned
  non-contiguous input. Cached device tests pass an explicit CUDA ordinal.
- Check storage independence, strides, aliases and writeback where relevant.
  Factory tests check shape, dtype, device and allocation without comparing
  uninitialized contents. Use small or deterministic inputs where possible.

The organization follows [LLVM's testing guidance](https://llvm.org/docs/TestingGuide.html)
for focused regression tests, and the named semantic case matrices used by
[Triton's language tests](https://github.com/triton-lang/triton/blob/main/python/test/unit/language/test_core.py).
The project retains `unittest` and uses `subTest` rather than adopting another
runner or importing either project's test infrastructure.

All e2e test methods must use `test.support.cuda.assert_no_cuda_leak`. Keep JIT
instances outside the callback and create temporary inputs and results inside it.
The callback checks the full case matrix and specialization counts. Two warmup
rounds precede ten measured rounds. The helper collects garbage, synchronizes
CUDA and records `torch.cuda.memory_allocated` at the baseline and after each
measured round.

Run these checks serially in the current process without `empty_cache` or reserved
memory comparisons. The assertion detects cached-execution leaks in
PyTorch-managed CUDA allocations, not CPU, native FFI objects or allocations
outside that allocator.

## Coverage

`unit/test_guards.py` covers guard parsing, supported expression IR, invalid
comparisons, tensor metadata, nested container-length extraction without an FX
importer, and the conservative `_base` fallback.
`unit/test_cleanup.py` checks attribute-free semantic resource construction,
borrowing low-level pure payloads, repeated and borrowed exports, Result branch
exports, independent bind arities and lexical input-cache reuse. Lit owns
reference ordering, mixed extraction, manual Base behavior and unsupported CFG
diagnostics. CUDA checks cover PyTorch-managed allocations, not CPU FFI objects.
`unit/test_importer.py` checks C ABI type spellings and Python type registration (including CAny),
semantic container bindings and construction, Result binding, payload type registration, parsed
OpView payload access, CABI-only Result validation, shared boxed Result type inference and boundary
boxing/widening, owned string Result boundaries, insertion-point restoration, empty argument construction,
explicit None positions and diagnostics for unsupported symbolic powers. The
e2e suites cover public JIT execution, container reconstruction, conditional
branches, symbolic shapes, specialization reuse, aliases and mutations. Container
execution tests also return borrowed inputs and repeated resource outputs, then
drop one alias and continue using another to check independent output references.
`e2e/test_local_call.py` executes local packed calls through the complete
lowering pipeline, covering direct and monadic success/failure, zero arguments,
Union/Str boundaries erased to CAny, failure-path output-slot gating, and payload
ownership with repeated CUDA tensor payloads.
`e2e/test_kernel_launch.py` covers mutation-only Triton launches, string
specializations and their None Result payloads; container tests include empty containers and consecutive None
outputs. Verifier diagnostics and Result lowering belong to the focused Lit
tests under `core/test/Dialect/TVMFFI/` and `core/test/Conversion/`.

Repeated calls exercise compiled execution as well as warmup. The unindexed
CUDA device spelling retains its value check without claiming cache reuse. Use
a fresh importer for each program and discard failed importers and input
tables. Test counts describe inventory, not a coverage target.

The following table maps the former flat test layout to current tests.

| Former test | Replacement | Preserved contract / parameters |
| --- | --- | --- |
| `test_all.py::test_all_with_list_dims` | `test/e2e/test_all.py::test_all_with_list_dims` | List dims [1, 0], keepdim=False, bool dtype and shape (11, 1). |
| `test_arithmetic.py::test_scalar_arithmetic` | `test/e2e/test_mul.py::test_scalar_overload; test/e2e/test_sub.py::test_scalar_and_tensor_overloads` | mul scalar 0.5; sub scalar 0.5 with alpha=2.0. |
| `test_arithmetic.py::test_tensor_sub` | `test/e2e/test_sub.py::test_scalar_and_tensor_overloads` | Tensor overload with alpha=1.0. |
| `test_clone.py::test_clone_contiguous_materializes_storage` | `test/e2e/test_clone.py::test_clone_memory_formats` | Non-contiguous input, contiguous independent allocation, input mutation isolation. |
| `test_clone.py::test_clone_preserve_format` | `test/e2e/test_clone.py::test_clone_memory_formats` | Non-contiguous strides and independent allocation. |
| `test_empty.py::test_call_empty` | `test/e2e/test_empty.py::test_empty_allocation` | Factory shape (3, 4), float32 and device; no uninitialized value comparison. |
| `test_empty.py::test_call_empty_like` | `test/e2e/test_empty_like.py::test_empty_like_allocation` | Input shape/dtype and independent allocation; oversized input removed. |
| `test_frontend.py::test_argument_specialization` | `test/e2e/test_specialization.py::test_argument_specialization` | Explicit CUDA/CPU devices, unindexed CUDA execution, float32/float16/reused dtype. |
| `test_frontend.py::test_cached_result_and_writeback` | `test/e2e/test_arange.py::test_cached_factory_result; test/e2e/test_add_.py::test_inplace_alias` | Packed Any tensor result and mutation writeback. |
| `test_frontend.py::test_container_inputs_unpack` | `test/e2e/test_containers.py::test_container_inputs_unpack` | List, nested list/tuple and tuple inputs. |
| `test_frontend.py::test_string_specialization` | `test/e2e/test_specialization.py::test_string_specialization` | Empty strings, 7/8-byte boundaries, long strings, nested results and specialization count. |
| `test_fx_importer.py::test_cond` | `test/e2e/test_cond.py::test_cond` | Both runtime predicate branches; sizes 3, 6 and 8. |
| `test_fx_importer.py::test_dynamic_shapes` | `test/e2e/test_reshape.py::test_dynamic_shapes` | Sizes 6, 10 and 14 reuse one specialization; runtime size/floordiv/mul IR. |
| `test_fx_importer.py::test_inplace_alias` | `test/e2e/test_add_.py::test_inplace_alias` | View return aliases input, with warmup and cached mutation writeback. |
| `test_fx_importer.py::test_multiple_aten_results` | `test/e2e/test_topk.py::test_multiple_aten_results` | Top-k values and indices on warmup/cached calls; deterministic distinct values. |
| `test_fx_importer.py::test_nested_outputs` | `test/e2e/test_containers.py::test_nested_outputs` | Dict/tuple/list pytree with tensor, integer and None leaves. |
| `test_fx_importer.py::test_symbolic_integer_power` | `test/e2e/test_pow.py::test_symbolic_integer_power` | Sizes 3, 5 and 7 reuse one specialization. |
| `test_guards.py::test_ast_lowering` | `test/unit/test_guards.py::GuardExpressionTest.test_supported_expressions` | All existing eight guard-expression cases, result types and supported native operations. |
| `test_guards.py::test_constant_match` | `test/unit/test_guards.py::GuardParsingTest.test_constant_values; GuardParsingTest.test_rejected_guards` | All existing seven supported and five rejected constant expressions. |
| `test_guards.py::test_guard_kinds` | `test/unit/test_guards.py::GuardParsingTest.test_supported_kinds` | Duplicate, ignored, sequence, shape-env and type-match guards. |
| `test_guards.py::test_rejected_guards_and_invariants` | `test/unit/test_guards.py::GuardParsingTest.test_rejected_guards; GuardParsingTest.test_invalid_device; GuardExpressionTest.test_invalid_comparisons; GuardExpressionTest.test_tensor_base_fallback` | All rejected guard cases, identity/comparison invariants, _base warning and invalid device. |
| `test_guards.py::test_tensor_match` | `test/unit/test_guards.py::GuardParsingTest.test_tensor_metadata` | Nested source path, type, dtype, device, requires_grad, rank and attribute absence. |
| `test_overwrite.py::test_copy_mutation` | `test/e2e/test_copy_.py::test_copy_writeback` | Public copy_ verifies destination writeback. |
| `test_t.py::test_call_t_rectangular_view` | `test/e2e/test_t.py::test_transpose_view` | Rectangular transpose shape, values, strides, shared storage and mutation visibility. |

## Lit consolidation

| Former coverage | Replacement / reason |
| --- | --- |
| Pipeline arithmetic, empty and transpose per-op dispatch | Representative `Pipeline/aten-dispatch.mlir` plus Python operator execution and storage semantics; generic packed calls and cached-handle lifetimes remain in `TVMFFIToLLVM/runtime-ops.mlir`. |
| Pipeline clone graph and two format wrappers | One representative dispatch wrapper; both formats retain numerical, stride and allocation checks through public JIT. |
| Pipeline all-dims, overwrite and tensor-literal duplicated graph bodies | One public FFI function per semantic case, preserving pipeline checks and runtime semantics. |
| GPU scalar get fixture | `TVMFFIToLLVM/scalar-extraction.mlir`, running only its owning conversion. |
| Combined GPU/ABI/LLVM and metadata tests | `Pipeline/kernel-launch.mlir` and `Pipeline/metadata.mlir`; the local GPU fixture checks semantic operands separately. |
| Legacy ownership pass tests | Automatic decomposition ownership tests and `unit/test_cleanup.py` cover reference lifetimes without attributes; CFG inference is unsupported. |
| Conversion bool/int pure printing | Existing dialect get/to tests; bool-to-int LLVM data flow remains in the conversion test. |
| Scalar min/max pure printing and unrelated arith folding | Pure printing and upstream-only folding checks are removed; no standalone scalar execution suite remains. |
| Full-pipeline ABI checks mixed with pass-local CFG tests | Dedicated `Pipeline/packed-abi.mlir`; local scalar/multi-result and CFG checks remain in `TVMFFIToFunc/abi-and-cfg.mlir`. |

Do not replace zero-initialization, lookup/call status or reference-count
contracts with broad wildcards when minimizing IR. Retain meaningful SSA
captures and reuse them for argument, result and ownership data flow.
