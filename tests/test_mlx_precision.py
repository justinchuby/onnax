# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Native MLX precision regressions; enable with ONNAX_TEST_MLX_EP=1."""

import os

import pytest
import test_softmax_precision as precision

from examples.huggingface.numerics import audit
from examples.lora.mlx_runtime import session

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in"
)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("margin", [18.0, 40.0, 80.0])
def test_native_softmax_derivatives(op, margin):
    precision.test_confident_class_retains_gradient(session, op, margin)


def test_native_softplus_derivatives():
    precision.test_softplus_negative_tail_and_hvp(session)


def test_native_kernel_audit():
    result = audit(mlx=True)
    for row in result["operators"].values():
        assert row["executed_providers"] == ["MLXExecutionProvider"]
        assert row["cpu_fallback_disabled"]
    assert result["operators"]["Sigmoid"]["ort_mlx"][0] > 0
    assert result["operators"]["Sigmoid"]["ort_mlx"][1] > 0
    assert result["operators"]["LogSoftmax"]["ort_mlx"][1] == -200
