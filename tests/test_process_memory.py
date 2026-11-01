# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Verify the kernel-counter ABI and explicit unavailable/error results."""

import ctypes
import sys
from types import SimpleNamespace

import pytest

from examples import process_memory


def test_rusage_v4_layout_matches_macos_sdk():
    # Cross-checked with sizeof/offsetof from the macOS SDK sys/resource.h.
    struct = process_memory._RUsageInfoV4
    assert ctypes.sizeof(struct) == 296
    assert struct.ri_phys_footprint.offset == 72
    assert struct.ri_lifetime_max_phys_footprint.offset == 240


def test_current_process_counters():
    result = process_memory.physical_footprint()
    assert result["available"] == (sys.platform == "darwin")
    if result["available"]:
        assert 0 < result["current_bytes"] <= result["lifetime_peak_bytes"]
        assert result["current_resident_bytes"] > 0
    else:
        assert "macOS" in result["reason"]


def test_unavailable_platform_does_not_load_native_library(monkeypatch):
    monkeypatch.setattr(process_memory, "sys", SimpleNamespace(platform="linux"))

    def unexpected(*args, **kwargs):
        raise AssertionError("Native library must not load")

    monkeypatch.setattr(process_memory.ctypes, "CDLL", unexpected)
    assert not process_memory.physical_footprint()["available"]


def test_failed_kernel_query_is_an_error(monkeypatch):
    class Function:
        def __call__(self, *args):
            return -1

    class Library:
        proc_pid_rusage = Function()

    monkeypatch.setattr(process_memory, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(process_memory.ctypes, "CDLL", lambda *a, **k: Library())
    monkeypatch.setattr(process_memory.ctypes, "get_errno", lambda: 13)
    with pytest.raises(OSError) as error:
        process_memory.physical_footprint()
    assert error.value.errno == 13


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel counters")
def test_child_process_counters():
    import subprocess
    import time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        result = process_memory.physical_footprint(child.pid)
        deadline = time.monotonic() + 1
        while result["current_bytes"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
            result = process_memory.physical_footprint(child.pid)
        assert result["available"]
        assert 0 < result["current_bytes"] <= result["lifetime_peak_bytes"]
    finally:
        child.terminate()
        child.wait(timeout=5)
