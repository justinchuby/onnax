# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Provider C ABI calls, byte preservation and native cache-release replay."""

import ctypes
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples import mlx_provider_memory
from examples.mlx_memory_lifecycle import equal_arrays, run


class Function:
    def __init__(self, value=0):
        self.value = value
        self.status = 0
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if args:
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_size_t))[0] = self.value
        return self.status


def test_bundled_c_abi_and_errors(monkeypatch):
    functions = {
        f"mlx_get_{n}_memory": Function(i)
        for i, n in enumerate(("active", "cache", "peak"), 1)
    }
    functions["mlx_clear_cache"] = Function()
    library = SimpleNamespace(**functions)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime_ep_mlx",
        SimpleNamespace(library_path=lambda: "/mock/provider.dylib"),
    )
    paths = []

    def load(path):
        paths.append(path)
        return library

    monkeypatch.setattr(mlx_provider_memory.ctypes, "CDLL", load)
    provider = mlx_provider_memory.ProviderMemory()
    assert paths == ["/mock/libmlxc.dylib"]
    assert provider.read() == {"active_bytes": 1, "cache_bytes": 2, "peak_bytes": 3}
    assert library.mlx_get_active_memory.argtypes == [ctypes.POINTER(ctypes.c_size_t)]
    assert library.mlx_get_active_memory.restype is ctypes.c_int
    provider.clear_cache()
    assert library.mlx_clear_cache.calls == 1
    library.mlx_clear_cache.status = 1
    with pytest.raises(RuntimeError, match="cache clear failed"):
        provider.clear_cache()
    library.mlx_get_cache_memory.status = 1
    with pytest.raises(RuntimeError, match="cache memory query failed"):
        provider.read()


def test_byte_comparison_detects_names_dtype_shape_and_signed_zero():
    expected = {"p": np.zeros(2, np.float32)}
    equal_arrays({"p": expected["p"].copy()}, expected)
    for actual in [
        {"q": expected["p"]},
        {"p": np.zeros(2, np.float64)},
        {"p": np.zeros((1, 2), np.float32)},
        {"p": np.array([-0.0, 0.0], np.float32)},
    ]:
        with pytest.raises(AssertionError):
            equal_arrays(actual, expected)


def test_byte_comparison_rejects_nonfinite_equal_values():
    for number in (float("nan"), float("inf")):
        values = {"p": np.array([number], np.float32)}
        with pytest.raises(AssertionError, match="Nonfinite"):
            equal_arrays(values, values)


@pytest.mark.parametrize("interval", [True, -1, 10001, float("inf"), float("nan"), "250"])
def test_invalid_settle_interval(interval):
    with pytest.raises(ValueError, match="finite settle interval"):
        run("unused", settle_ms=interval)


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_release_and_exact_resume(tmp_path, precision):
    report = run(tmp_path, precision=precision, microbatch=4096)
    assert report["resumed_outputs_byte_equal"] and report["preserved_input_checkpoint"]
    assert report["resumed_step"] == 111
    assert len(report["compared_outputs_and_state"]) == 18
    phases = {p["phase"]: p for p in report["phases"]}
    assert phases["before_release"]["provider"]["cache_bytes"] > 0
    assert phases["after_provider_settle"]["provider"]["cache_bytes"] == 0
    assert phases["resumed"]["provider"]["cache_bytes"] > 0
    assert report["baseline_samples"] == 100
