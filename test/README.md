<!-- Part of the Trident project, under the MIT License. -->
<!-- SPDX-License-Identifier: MIT -->

# Python tests

Use ordinary `unittest` discovery and named `subTest` cases. Tests are grouped by
capability, with no generated methods, discovery hooks or compatibility modules.

| Directory | Responsibility | Prerequisites |
| --- | --- | --- |
| `unit` | Python bindings, frontend construction, guard parsing and diagnostics | Installed native extension; no GPU execution |
| `e2e` | Public JIT behavior and cached execution; local packed-call integration | CUDA |
| `support` | Fresh IR contexts and CUDA allocation checks | Helpers only |

## Running

From the repository root with the editable build installed:

```bash
python -m unittest discover -s test -p 'test_*.py'
python -m unittest discover -s test/unit -t . -p 'test_*.py'
python -m unittest discover -s test/e2e -t . -p 'test_*.py'
python -m unittest test.e2e.test_storage
cmake --build build/core --target check-trident-core
```

CUDA execution tests require a usable GPU; selecting `unit` does not run them.

## Test selection

Follow [LLVM’s testing guide](https://llvm.org/docs/TestingGuide.html): keep
small regression inputs at the implementation layer and reserve execution tests
for observable integration behavior. Compression means removing equivalent
paths, not combining unrelated regressions into one large program. Preserve
separate cases when they select different branches or have different failure
contracts.

For example, one transposed-view mutation case checks layout, input writeback
and bidirectional aliasing; one nested output pytree includes empty containers
and None leaves. A wholly empty output remains separate because it takes the
no-result execution path. Copy writeback remains separate because its source
and destination can be reversed independently of view aliasing.

## Coverage ownership

| Suite | Contract |
| --- | --- |
| `unit/test_guards.py` | Guard parsing, expression IR, metadata, nested input extraction and conservative fallbacks |
| `unit/test_bindings.py` | Python type construction, registration, parameter access, operation properties and Python construction errors |
| `unit/test_importer.py` | Semantic frontend and scalar values, argument/result construction, bind scopes, uncached input extraction and importer diagnostics |
| `e2e/test_operators.py` | Scalar/tensor overloads, schema arguments, reductions and multiple results |
| `e2e/test_factories.py` | No-argument calls, factory metadata and independent allocations |
| `e2e/test_storage.py` | Clone formats, view strides, aliases and mutation writeback |
| `e2e/test_dynamic.py` | Runtime shapes, symbolic arithmetic and typed scalar outputs, conditional branches and specialization reuse |
| `e2e/test_containers.py` | Input unpacking, output pytrees, None and independent output references |
| `e2e/test_specialization.py` | Device, dtype and string specialization, including string storage boundaries and Unicode |
| `e2e/test_kernel_launch.py` | Triton execution, specialization rejection and cleanup |
| `e2e/test_regions.py` | Packed region execution, zero/multiple iterations, replacement/identity payloads and loop allocation cleanup |
| `e2e/test_local_call.py` | Executable packed-call success/failure, erased boundaries and tensor payload ownership |

Lit owns textual parsing/verifier diagnostics, pass-local lowering, reference
ordering and representative cross-pass integration. Python binding construction
errors remain Python tests. Avoid repeating lowering assertions in Python or
whole-pipeline ABI fixtures for every operator using generic dispatch. Python
tests must not load Lit files as executable fixtures.

Generic type constraints use representative payload categories rather than every
scalar width or object type. Arithmetic dispatch keeps scalar and tensor overloads
with schema keyword arguments; equivalent operator names need no separate case.

## Writing tests

- Add cases to the capability that owns the contract. Use local case tables and
  named `subTest` cases for equivalent flows; keep aliasing and mutation checks
  explicit. Each distinct layout, branch or diagnostic needs a representative.
- Each test owns its JIT instances and IR context. Do not compile at module
  import or share mutable specialization state between tests.
- Check initial and repeated execution and assert specialization counts where
  cache reuse is required. Use owned inputs to avoid conservative `_base` guard
  misses; preserve-format clone creates an owned non-contiguous input. Device
  reuse tests pass an explicit CUDA ordinal. The unindexed CUDA spelling is
  checked separately without claiming reuse.
- Check storage independence, strides, aliases and writeback where relevant.
  Factory tests compare shape, dtype, device and allocations, never uninitialized
  contents. Use small deterministic inputs where practical.
- All e2e methods use `test.support.cuda.assert_no_cuda_leak`. Keep JIT instances
  outside the callback and temporary inputs/results inside it. The callback
  checks its case matrix and specialization counts. Two warmup rounds precede
  ten measured rounds; garbage collection and CUDA synchronization must restore
  `torch.cuda.memory_allocated` to the baseline after every measured round.

Run allocation checks serially in the current process without `empty_cache` or
reserved-memory comparisons. They detect cached-execution leaks in PyTorch's
CUDA allocator, not CPU, native FFI objects or external allocations. Keep the
lookup/call status, output-slot gating and reference-count data flow explicit in
Lit; broad wildcards are not a substitute for those contracts.
