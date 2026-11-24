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
- LLVM/MLIR can be supplied from a local installation or fetched and built as an
  external project.
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
	-Ccmake.define.MLIR_ROOT=/path/to/llvm-install \
	-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Set `MLIR_ROOT` to use an existing LLVM/MLIR installation. CMake searches for
MLIR and fetches and builds the pinned revision if none is found.

Alternatively, set the `LLVM_SYSPATH` environment variable used by Triton. It
is used as the MLIR package root when `MLIR_ROOT` is not set:

```bash
LLVM_SYSPATH=/path/to/llvm-install \
	CC=clang CXX=clang++ uv pip install -ve . --no-build-isolation
```

The same CMake variable can be passed to pip without adding an environment
variable:

```bash
CC=clang CXX=clang++ python -m pip install . \
	--config-settings=build-dir=build \
	--config-settings=cmake.define.MLIR_ROOT=/path/to/llvm-install
```

### Build Wheel

```bash
CC=clang CXX=clang++ uv build
```

To use a local LLVM/MLIR installation for a wheel build, pass the same backend
config setting, for example:

```bash
CC=clang CXX=clang++ uv build \
	-Ccmake.define.MLIR_ROOT=/path/to/llvm-install
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

## Pre-commit Checks

Install pre-commit and enable the Git hook from the repository root:

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

The hooks use pinned Ruff and clang-format versions in isolated environments.
Ruff checks Python with the basic `E4/E7/E9/F` lint rules, applies safe fixes,
and formats Python. clang-format formats C/C++ sources and headers using LLVM
style. TableGen (`.td`) files, build directories and generated outputs are
excluded. No project build or CUDA installation is needed to run these checks.

On commit, only staged files are checked. If a hook modifies files or reports
remaining errors, the commit stops; review the changes, fix any remaining
errors, and stage the files again before retrying the commit.

The existing `bash ./tools/format.sh` command also formats CMake files; CMake
formatting is not part of the pre-commit hooks.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Source files in this repository use MIT SPDX headers (`SPDX-License-Identifier: MIT`).

## Notes

- GPU/CUDA availability is expected by the provided examples/tests.
- Initial builds can take a long time because LLVM/MLIR is
	built/fetched as dependencies.
