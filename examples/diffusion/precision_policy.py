# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Norm budgets for the diffusion example's projection compute formats."""

import math


def projection_norm_rtol(precision):
    """Return FP32 tolerance or one half-format epsilon for whole-path norms."""
    values = {"fp32": 1e-4, "bf16": 2**-7, "fp16": 2**-10}
    if precision not in values:
        raise ValueError("Select fp32, bf16 or fp16 projection precision")
    return values[precision]


def mse_difference_bound(reference_loss, prediction_rms_difference):
    """Cauchy bound for the MSE change caused by an observed prediction delta."""
    if (
        reference_loss < 0
        or prediction_rms_difference < 0
        or not all(
            math.isfinite(value) for value in (reference_loss, prediction_rms_difference)
        )
    ):
        raise ValueError("Expected finite nonnegative MSE and prediction RMS difference")
    return (
        2 * math.sqrt(reference_loss) * prediction_rms_difference
        + prediction_rms_difference**2
    )
