# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from trident import capi_utils


def _stub_common_libraries(monkeypatch):
    monkeypatch.setattr(capi_utils, "find_ffi_exception_library", lambda: "ffi")
    monkeypatch.setattr(capi_utils, "find_aten_ffi_library", lambda: "aten")
    monkeypatch.setattr(
        capi_utils, "torch", SimpleNamespace(_C=SimpleNamespace(__file__="torch"))
    )
    monkeypatch.setattr(
        capi_utils,
        "tvm_ffi",
        SimpleNamespace(
            libinfo=SimpleNamespace(find_libtvm_ffi=lambda: "tvm_ffi")
        ),
    )


def test_cuda_runtime_libraries_unchanged(monkeypatch):
    _stub_common_libraries(monkeypatch)
    monkeypatch.setattr(capi_utils, "FLAGTREE_BACKEND", "nvidia")
    monkeypatch.setattr(
        capi_utils, "find_mlir_cuda_runtime_library", lambda: "mlir_cuda"
    )
    monkeypatch.setattr(
        capi_utils,
        "find_ascend_runtime_library",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Ascend lookup")),
    )

    assert capi_utils.find_runtime_libraries() == [
        "ffi",
        "aten",
        "mlir_cuda",
        "torch",
        "tvm_ffi",
    ]


def test_ascend_runtime_replaces_cuda_runtime(monkeypatch):
    _stub_common_libraries(monkeypatch)
    monkeypatch.setattr(capi_utils, "FLAGTREE_BACKEND", "ascend")
    monkeypatch.setattr(
        capi_utils, "find_ascend_runtime_library", lambda: "ascend"
    )
    monkeypatch.setattr(
        capi_utils,
        "find_mlir_cuda_runtime_library",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected CUDA lookup")),
    )

    assert capi_utils.find_runtime_libraries() == [
        "ffi",
        "aten",
        "ascend",
        "torch",
        "tvm_ffi",
    ]
