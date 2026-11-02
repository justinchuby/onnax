# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Verify and summarize independent full/bounded decoder measurements."""

import argparse
import json

import numpy as np


def compare(full, bounded):
    with open(full, encoding="utf-8") as stream:
        left = json.load(stream)
    with open(bounded, encoding="utf-8") as stream:
        right = json.load(stream)
    for field in (
        "input_sha256",
        "library_sha256",
        "dtype",
        "threads",
        "ort_graph_optimization",
        "runtime_environment",
        "versions",
        "hardware",
        "source_sha256",
    ):
        if left.get(field) != right.get(field):
            raise ValueError(f"Measurement configuration differs: {field}")
    for field in (
        "family",
        "real",
        "provider",
        "sequence",
        "valid_tokens",
        "warmup",
        "samples",
    ):
        if left["options"][field] != right["options"][field]:
            raise ValueError(f"Workload differs: {field}")
    if (
        left["options"]["head_chunk_size"] is not None
        or right["options"]["head_chunk_size"] is None
    ):
        raise ValueError("Expected full-head control followed by bounded-head measurement")
    errors = {}
    with np.load(full + ".npz") as a, np.load(bounded + ".npz") as b:
        if set(a.files) != set(b.files) or "loss" not in a.files:
            raise ValueError("Loss/adapter gradient output mapping differs")
        for name in a.files:
            if not np.isfinite(a[name]).all() or not np.isfinite(b[name]).all():
                raise ValueError(f"Nonfinite numerical output: {name}")
            if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape:
                raise ValueError(f"Gradient dtype/shape differs: {name}")
            np.testing.assert_allclose(
                a[name], b[name], rtol=1e-4, atol=1e-6, err_msg=f"Full/bounded output: {name}"
            )
            errors[name] = float(np.max(np.abs(a[name] - b[name])))
    return {
        "version": 1,
        "full": full,
        "bounded": bounded,
        "correctness": {"rtol": 1e-4, "atol": 1e-6, "max_abs_errors": errors},
        "timings": {"full": left["timings"], "bounded": right["timings"]},
        "peak_rss_bytes": {"full": left["peak_rss_bytes"], "bounded": right["peak_rss_bytes"]},
        "timing_scope": left["timing_scope"],
        "memory_scope": left["memory_scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True)
    parser.add_argument("--bounded", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.full, args.bounded)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report["timings"]))


if __name__ == "__main__":
    main()
