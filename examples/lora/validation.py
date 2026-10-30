# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Numerical comparisons shared by workload evaluations."""

import numpy as np


def discrepancy(actual, expected, *, rtol=1e-4, atol=1e-6):
    """Record the unchanged FP32 acceptance tolerance alongside observed errors."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("Comparison shapes must match")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("Comparison requires finite values")
    bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result = {"mismatched": int(np.count_nonzero(bad)), "max_abs": float(np.max(error))}
    if np.any(bad):
        index = tuple(np.argwhere(bad)[0])
        result["first"] = {
            "index": [int(i) for i in index],
            "actual": float(actual[index]),
            "expected": float(expected[index]),
        }
    return result
