# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exercise independent training-quality comparisons and paired initialization."""

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
pytest.importorskip("torch")


def test_independent_quality_runs_share_initialization_and_data():
    from examples.huggingface.training_impact import run

    report = run(
        engines=("torch_cpu", "onnx_reference"),
        seeds=(17, 29),
        steps=2,
        interval=1,
        microbatch=2,
        accumulation=2,
    )
    assert report["frozen_unchanged"]
    assert report["train_lines"] > 0 and report["heldout_lines"] > 0
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        assert len(trial["training_loss"]) == 2
        assert trial["final_training_loss"] < trial["initial_training_loss"]
        assert np.isfinite(trial["final_validation_loss"])
    for comparison in report["comparisons"]:
        assert comparison["gradient_snapshots"][0]["cosine"] > 0.9999
        assert np.isfinite(comparison["parameters"]["relative_l2"])
    assert len(report["validation_summary"]["torch_cpu"]["values"]) == 2


def test_gradient_direction_distinguishes_scale_and_sign():
    from examples.huggingface.training_impact import _vector_metrics

    x = np.array([1, 2, 3], np.float32)
    assert _vector_metrics(x, x * 2)["cosine"] == pytest.approx(1)
    assert _vector_metrics(x, -x)["cosine"] == pytest.approx(-1)
    assert _vector_metrics(x, x * 2)["relative_l2"] == pytest.approx(0.5)
    assert _vector_metrics(np.zeros_like(x), x)["cosine"] == 0
