# Trident

Trident is an experimental bridge that compiles Python functions containing
Triton/Torch-style compute into MLIR/LLVM, then executes through tvm_ffi.

The repository contains:

- A C++/MLIR core (`core`) built with CMake.
- A Python package (`trident`) with `jit` and `compile` entry points.
- A lightweight FFI runtime layer (`ffi`) providing composable error types.
- End-to-end examples and tests for ATen op lowering/execution.

## Current Status

- Build system is wired through `scikit-build-core` + CMake.
- LLVM/MLIR is fetched and built as an external project.
- Python extension artifacts are installed directly under `trident/` in wheel/editable builds.

## Repository Layout

- `core`: MLIR/C++ implementation, dialects, passes, runtime glue.
- `examples`: Functional demos (`add.py`, `attn.py`, `mm.py`, `softmax.py`).
- `ffi`: Lightweight C++ shared library for FFI-level types (`trident.ffi.Exception`).
- `python/trident`: Python APIs (`jit`, `compile`) and runtime backend.
- `test`: Python guard unit tests and public-JIT end-to-end tests.

## Prerequisites

- CMake 3.29+
- CUDA Toolkit (for GPU execution)
- Clang/Clang++
- Linux
- Ninja
- Python 3.10+
- `uv` (recommended Python package/build workflow)

Dependencies as declared in `pyproject.toml`:

**Runtime** (also listed under `[project] dependencies`):
- `apache-tvm-ffi`
- `torch>=2.10.0`

**Build-time** (also required at runtime for `torch`):
- `apache-tvm-ffi`
- `scikit-build-core`, `nanobind`, `cmake`, `ninja`
- `torch>=2.10.0`
- `triton`

## Build And Install

### Editable Debug Build (recommended during development)

```bash
CC=clang CXX=clang++ uv pip install -ve . \
	--no-build-isolation \
	-Cbuild-dir=build \
	-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Build Wheel

```bash
CC=clang CXX=clang++ uv build
```

## Run Examples

After install, run from repository root:

```bash
python examples/*.py
```

All examples compare direct Torch/Triton outputs with Trident-jitted outputs.

## Run Tests

```bash
python -m unittest discover -s test -p "test_*.py"
cmake --build build/core --target check-trident-core
```

Python tests are organized into `test/unit` (Python bindings, guard expression IR
and frontend construction) and `test/e2e` (public `trident.jit` behavior).
End-to-end execution tests require a usable CUDA GPU. Unit tests do not require GPU execution.

Run a group or a focused module with:

```bash
python -m unittest discover -s test/unit -t . -p "test_*.py"
python -m unittest test.e2e.test_specialization
```

See [the Python test guide](test/README.md) for capability groups and coverage
boundaries, and [the core test guide](core/test/README.md) for Lit.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Source files in this repository use MIT SPDX headers (`SPDX-License-Identifier: MIT`).

## Notes

- GPU/CUDA availability is expected by the provided examples/tests.
- Initial builds can take a long time because LLVM/MLIR is
	built/fetched as dependencies.
