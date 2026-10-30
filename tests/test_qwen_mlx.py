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
