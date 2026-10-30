# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Opt-in native MLX Qwen forward, gradients and independent updates."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
def test_nf4_qwen_native_training():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.mlx_validation import run

    run(library=os.environ.get("ONNAX_MLX_EP_LIBRARY"))


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
def test_native_same_input_layer_audit():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.layer_audit import run

    report = run(mlx=True, library=os.environ.get("ONNAX_MLX_EP_LIBRARY"))
    assert report["execution_provider"] == "MLXExecutionProvider"
    assert report["cpu_fallback_disabled"]
    assert report["logits"]["relative_l2"] < 1e-4
    for op in ("RMSNormalization", "Swish"):
        assert report["operator_summary"][op]["violations"] == 0
    dots = [row["worst_scaled_dot"] for row in report["operators"] if row["op"] == "MatMul"]
    assert dots and all("ort_mlx" in dot and "ort_cpu" not in dot for dot in dots)
