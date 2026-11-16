# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import importlib.metadata as im

import torch
import tvm_ffi


def find_runtime_libraries() -> list[str]:
    """Find the Trident runtime libraries.

    Returns:
        Paths to the Trident runtime libraries

    Raises:
        RuntimeError: If any runtime library is not found
    """
    torch_library: str | None = torch._C.__file__
    assert torch_library is not None, "PyTorch C extension has no library path"
    package_dir = (im.distribution("trident")._path.parent / "trident").resolve()
    libraries: list[str] = []
    # Load the FFI runtime before its generated ATen wrappers.
    for filename, description in (
        ("libTridentFFI.so", "Trident FFI Exception"),
        ("libTridentAtenFFI.so", "Trident ATen FFI"),
        ("libmlir_cuda_runtime.so", "MLIR CUDA runtime"),
    ):
        library = package_dir / "lib" / filename
        assert library.is_file(), f"missing {description} library: {library}"
        libraries.append(f"{library}")
    return [
        *libraries,
        torch_library,
        tvm_ffi.libinfo.find_libtvm_ffi(),
    ]
