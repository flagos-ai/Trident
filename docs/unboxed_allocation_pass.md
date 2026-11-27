# Trident Host Path Optimizations (Direct Allocations/Views & Compiler Lowering)

Trident enables a unified set of host-side lowering and runtime optimizations by
default inside `trident-lowering-pipeline` and `TridentGraphModule`:

1. **Direct C ABI for allocations and views (`direct-allocations` + `flatten-direct-allocations`)**:
   - Rewrites Result-producing `aten::empty.memory_format`, `aten::empty_strided`,
     `aten::empty_like`, `aten::zeros`, `aten::zeros_like`, and
     `aten::view` / `aten::_unsafe_view` into `DirectAllocationOp`.
   - Flattens local shape/stride arrays into stack-allocated `int64_t` buffers,
     avoiding `ffi.Array` heap allocations.
   - Lowers common signatures to compact register-only C ABIs (`<= 6` arguments):
     `trident_c_allocate_empty`, `trident_c_allocate_empty_like`,
     `trident_c_allocate_zeros`, `trident_c_allocate_zeros_like`, and
     `trident_c_view` (with zero-copy `IncRef` short-circuit on no-op shapes),
     falling back to `trident_c_allocate` for explicit layout/memory_format or
     dynamic array handles.
   - Wraps returned `at::Tensor` instances directly in `ATenTensorObj` (a
     `tvm::ffi::TensorObj` subclass) to eliminate intermediate `at::toDLPack()`
     `ATenDLMTensor` heap allocations while preserving standard `DLTensor` ABI
     and public `at::*` unboxed dispatcher semantics.

2. **Unboxed `scf.if` Result tuples & native scalar equality (`unbox-scf-if-tuples`)**:
   - Unboxes `tvm_ffi_api.tuple` results across `scf.if` boundaries so Monad
     Results and `KernelLaunchOp` outcomes stay in SSA registers instead of
     allocating `ffi.Array` and calling `ffi.ArrayGetItem`.
   - Lowers primitive scalar `tvm_ffi.eq` (`int`/`bool`/`float`/`none`) to native
     `arith` comparisons so Triton `tl.constexpr` guards fold at compile time
     instead of calling `ffi.StructuralEqual`.

3. **Compile-time `torch.dtype` folding & specialized Python fast call**:
   - `importer.py` folds constant `torch.dtype` arguments to integer `ScalarType`
     constants at import time instead of emitting runtime `tvm_ffi_to_torch_type`
     FFI calls.
   - `backend.py` generates arity-specialized positional fast wrappers for flat
     signatures with single-leaf outputs, bypassing `kwargs_wrapper`,
     `_map_containers`, and `pytree.tree_unflatten` on the warm path while
     preserving full fallback behavior.
