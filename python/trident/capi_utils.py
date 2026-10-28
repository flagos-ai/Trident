# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import functools
import importlib
import importlib.metadata as im
import pathlib

import torch
import tvm_ffi
from triton._flagtree_backend import FLAGTREE_BACKEND


@functools.cache
def _trident_pkg_dir() -> pathlib.Path:
    """Return the installed ``trident`` package directory.

    Uses ``importlib.metadata`` to locate the package root via the
    distribution RECORD — the same approach :mod:`tvm_ffi.libinfo`
    uses to find its own shared libraries.
    """
    return (im.distribution("trident")._path.parent / "trident").resolve()


def find_aten_ffi_library() -> str:
    """Find the generated ATen TVM FFI wrapper library."""
    aten_ffi_lib: pathlib.Path = _trident_pkg_dir() / "lib" / "libTridentAtenFFI.so"
    assert aten_ffi_lib.is_file(), f"missing Trident ATen FFI library: {aten_ffi_lib}"
    return f"{aten_ffi_lib}"


def find_ffi_exception_library() -> str:
    """Find the Trident FFI library.

    Returns:
        Path to ``libTridentFFI.so``

    Raises:
        RuntimeError: If the library is not found
    """
    ffi_exception_lib: pathlib.Path = _trident_pkg_dir() / "lib" / "libTridentFFI.so"
    assert ffi_exception_lib.is_file(), (
        f"missing Trident FFI Exception library: {ffi_exception_lib}"
    )
    return f"{ffi_exception_lib}"


def find_mlir_cuda_runtime_library() -> str:
    """Find the MLIR CUDA runtime library.

    Returns:
        Path to `libmlir_cuda_runtime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    cuda_runtime_lib: pathlib.Path = (
        _trident_pkg_dir() / "lib" / "libmlir_cuda_runtime.so"
    )
    assert cuda_runtime_lib.is_file(), (
        f"missing MLIR CUDA runtime library: {cuda_runtime_lib}"
    )
    return f"{cuda_runtime_lib}"


def find_ascend_runtime_library() -> str:
    """Find the Trident torch_npu/CANN launch runtime."""
    importlib.import_module("torch_npu")
    runtime = _trident_pkg_dir() / "lib" / "libTridentAscendRuntime.so"
    assert runtime.is_file(), f"missing Trident Ascend runtime library: {runtime}"
    return f"{runtime}"


def find_runtime_libraries() -> list[str]:
    """Find the Trident runtime libraries.

    Returns:
        Paths to the Trident runtime libraries

    Raises:
        RuntimeError: If any runtime library is not found
    """
    torch_library: str | None = torch._C.__file__
    assert torch_library is not None, "PyTorch C extension has no library path"
    libraries = [
        find_ffi_exception_library(),
        find_aten_ffi_library(),
    ]
    libraries.append(
        find_ascend_runtime_library()
        if FLAGTREE_BACKEND == "ascend"
        else find_mlir_cuda_runtime_library()
    )
    libraries.extend([torch_library, tvm_ffi.libinfo.find_libtvm_ffi()])
    return libraries
