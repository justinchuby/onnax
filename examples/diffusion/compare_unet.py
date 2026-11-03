# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare matched UNet benchmark artifacts and their captured derivatives."""

import argparse
import json
import math

import numpy as np

from examples.diffusion.precision_policy import projection_norm_rtol
from examples.huggingface.block_runtime import digest


def compare(baseline, candidate):
    """Require matching workload/state and validate every captured adapter gradient."""
    with open(baseline, encoding="utf-8") as stream:
        left = json.load(stream)
    with open(candidate, encoding="utf-8") as stream:
        right = json.load(stream)
    precision = left.get("projection_precision", "fp32")
    if precision != right.get("projection_precision", "fp32"):
        raise ValueError("Benchmark workload differs: projection_precision")
    tolerance = projection_norm_rtol(precision)
    for key in (
        "model",
        "revision",
        "export_identity",
        "adapter_state",
        "data",
        "dtype",
        "latent_shape",
        "conditioning_shape",
        "learning_rate",
        "optimizer",
        "optimizer_epsilon",
        "warmup",
        "samples",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Benchmark workload differs: {key}")
    for report in (left, right):
        if (
            report["samples"] < 1
            or report["warmup"] < 0
            or report["full_step_seconds"]["samples"] != report["samples"]
        ):
            raise ValueError("Incomplete benchmark sample count")
    checks = {}
    gradient_error_squared = gradient_norm_squared = 0.0
    with np.load(baseline + ".npz") as expected, np.load(candidate + ".npz") as actual:
        required = {"loss", "prediction", *left["adapter_state"]}
        if set(expected.files) != required or set(actual.files) != required:
            raise ValueError("Captured prediction/loss/adapter names differ")
        for name in expected.files:
            a, b = actual[name], expected[name]
            if (
                a.dtype != b.dtype
                or a.shape != b.shape
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
            ):
                raise AssertionError(f"Invalid captured derivative: {name}")
            if name in left["adapter_state"]:
                specification = left["adapter_state"][name]
                if (
                    list(a.shape) != specification["shape"]
                    or str(a.dtype) != specification["dtype"]
                ):
                    raise AssertionError(f"Gradient differs from parameter interface: {name}")
            error = a - b
            norm = float(np.linalg.norm(error))
            reference = float(np.linalg.norm(b))
            if name in left["adapter_state"]:
                gradient_error_squared += norm**2
                gradient_norm_squared += reference**2
            if precision == "fp32" or name in ("prediction", "loss"):
                budget = 1e-4 if name == "loss" else tolerance
                if norm > 1e-6 + budget * reference:
                    raise AssertionError(f"Tensor norm mismatch: {name}")
            if name == "loss" or (precision == "fp32" and name != "prediction"):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6, err_msg=name)
            checks[name] = dict(
                max_absolute_error=float(np.abs(error).max()),
                relative_l2_error=norm / max(reference, 1e-30),
                pointwise_tolerance_failures=int(
                    (np.abs(error) > 1e-6 + 1e-4 * np.abs(b)).sum()
                ),
            )
    gradient_error = math.sqrt(gradient_error_squared)
    gradient_norm = math.sqrt(gradient_norm_squared)
    if precision != "fp32" and gradient_error > 1e-6 + tolerance * gradient_norm:
        raise AssertionError("Combined gradient norm mismatch")
    return dict(
        projection_precision=precision,
        combined_gradient_error=gradient_error,
        combined_gradient_reference_norm=gradient_norm,
        norm_rtol=tolerance,
        pointwise_diagnostic_rtol=1e-4,
        model=left["model"],
        revision=left["revision"],
        latent_shape=left["latent_shape"],
        conditioning_shape=left["conditioning_shape"],
        captured_tensors=len(checks),
        checks=checks,
        baseline=dict(
            path=baseline,
            report_sha256=digest(baseline),
            capture_sha256=digest(baseline + ".npz"),
            provider=left["provider"],
            full_step_seconds=left["full_step_seconds"],
            physical_footprint=left["physical_footprint"],
            peak_process_rss_bytes=left["peak_process_rss_bytes"],
        ),
        candidate=dict(
            path=candidate,
            report_sha256=digest(candidate),
            capture_sha256=digest(candidate + ".npz"),
            provider=right["provider"],
            full_step_seconds=right["full_step_seconds"],
            physical_footprint=right["physical_footprint"],
            peak_process_rss_bytes=right["peak_process_rss_bytes"],
        ),
        candidate_to_baseline_median_ratio=right["full_step_seconds"]["median"]
        / left["full_step_seconds"]["median"],
        memory_scope="Isolated whole-process lifetime peak, including construction and runtime allocator caches; cache policies are recorded in source reports",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("captured_tensors", "candidate_to_baseline_median_ratio")
            }
        )
    )


if __name__ == "__main__":
    main()
