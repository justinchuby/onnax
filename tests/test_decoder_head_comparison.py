# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Performance comparisons require matching work and complete finite outputs."""

import json

import numpy as np
import pytest

from examples.huggingface.decoder_head_comparison import compare


@pytest.fixture
def measurements(tmp_path):
    paths = [str(tmp_path / name) for name in ("full.json", "bounded.json")]
    for path, chunk in zip(paths, (None, 16), strict=True):
        record = {
            "version": 1,
            "options": {
                "family": "qwen35",
                "real": False,
                "provider": "mlx",
                "sequence": 64,
                "valid_tokens": 32,
                "warmup": 10,
                "samples": 2,
                "head_chunk_size": chunk,
            },
            "input_sha256": "same-inputs",
            "source_sha256": {"model.py": "same-source"},
            "samples": [{"step_seconds": 0.1}, {"step_seconds": 0.12}],
            "timings": {"step_seconds": {"median": 0.11, "p95": 0.119}},
            "peak_rss_bytes": 1024,
            "timing_scope": "forward and backward",
            "memory_scope": "whole process",
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        np.savez(
            path + ".npz",
            loss=np.array(1, np.float32),
            **{"adapter.A": np.array([0.01, -0.02], np.float32)},
        )
    return paths


def test_compare_matching_results(measurements):
    result = compare(*measurements)
    assert result["correctness"]["max_abs_errors"] == {"loss": 0.0, "adapter.A": 0.0}


@pytest.mark.parametrize(
    "change", ["input", "source", "tokens", "scope", "partial", "nan_time"]
)
def test_reject_different_or_incomplete_work(measurements, change):
    path = measurements[1]
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
    if change == "input":
        record["input_sha256"] = "different-inputs"
    elif change == "source":
        record["source_sha256"]["model.py"] = "different-source"
    elif change == "tokens":
        record["options"]["valid_tokens"] -= 1
    elif change == "scope":
        record["timing_scope"] = "forward only"
    elif change == "partial":
        record["samples"].pop()
    else:
        record["samples"][0]["step_seconds"] = float("nan")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
    with pytest.raises(ValueError):
        compare(*measurements)


@pytest.mark.parametrize(
    "change", ["wrong_gradient", "nan_gradient", "missing_gradient", "dtype"]
)
def test_reject_incorrect_numerical_outputs(measurements, change):
    gradient = np.array([0.01, -0.02], np.float32)
    if change == "wrong_gradient":
        gradient[0] = 0.02
    elif change == "nan_gradient":
        gradient[0] = np.nan
    elif change == "dtype":
        gradient = gradient.astype(np.float64)
    arrays = {"loss": np.array(1, np.float32)}
    if change != "missing_gradient":
        arrays["adapter.A"] = gradient
    np.savez(measurements[1] + ".npz", **arrays)
    with pytest.raises((ValueError, AssertionError)):
        compare(*measurements)
