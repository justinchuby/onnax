# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Dimension-independent LoRA exports preserve gradients through frozen heads."""

import copy
import os

import numpy as np
import onnx_ir as ir
import pytest

from examples.lora.binding import CpuGradientBinding
from examples.lora.model import PARAMETERS, adapt, base_model, export, validate_interface
from examples.lora.train import reference, session
from examples.lora.validation import discrepancy


def test_nondefault_dimensions_export_and_gradients(tmp_path):
    rng = np.random.default_rng(42)
    w = rng.normal(size=(5, 7)).astype(np.float32)
    c = rng.normal(size=(7, 3)).astype(np.float32)
    base = base_model(w, c)
    before = ir.to_proto(base).SerializeToString()
    ir.save(base, tmp_path / "base.onnx")
    _, _, gradient, interface = export(tmp_path / "base.onnx", tmp_path)
    validate_interface(gradient, interface)
    assert interface["data_shapes"] == {"features": [None, 5], "target": [None, 3]}
    assert [row["shape"] for row in interface["parameters"]] == [[5, 4], [4, 7]]
    params = {
        name: rng.normal(0, 0.1, row["shape"]).astype(np.float32)
        for name, row in zip(PARAMETERS, interface["parameters"], strict=True)
    }
    executor = session(tmp_path / "gradient.onnx")
    bound = CpuGradientBinding(executor, interface)
    for batch in (1, 6):
        x = rng.normal(size=(batch, 5)).astype(np.float32)
        y = rng.normal(size=(batch, 3)).astype(np.float32)
        actual = executor.run(None, {"features": x, "target": y, **params})
        prediction, loss, grads = reference(x, y, params, w, c)
        bound.bind(x, y, params)
        bound_loss, bound_grads = bound.run()
        np.testing.assert_allclose(bound_loss, loss, rtol=1e-4, atol=1e-6)
        for name in PARAMETERS:
            np.testing.assert_allclose(bound_grads[name], grads[name], rtol=1e-4, atol=1e-6)
        for observed, expected in zip(
            actual, [prediction, loss, *grads.values()], strict=True
        ):
            np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-6)
    for shape in ([None, 0], [None, True], [None, 6], [2, 5]):
        bad = copy.deepcopy(interface)
        bad["data_shapes"]["features"] = shape
        with pytest.raises(ValueError, match="shape"):
            CpuGradientBinding(executor, bad)
    assert ir.to_proto(base).SerializeToString() == before
    ranked = adapt(base, rank=2)
    assert [list(v.shape) for v in ranked.graph.inputs[1:]] == [[5, 2], [2, 7]]


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        adapt(base_model(np.ones((5, 7), np.float32), np.ones((7, 3), np.float32)), rank=rank)


def test_discrepancy_preserves_tolerance_and_rejects_nonfinite():
    result = discrepancy(np.array([0.0, 2.0]), np.array([2e-6, 2.0]))
    assert result["mismatched"] == 1
    assert result["first"]["index"] == [0]
    with pytest.raises(ValueError, match="finite"):
        discrepancy(np.array([np.nan]), np.array([0.0]))


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1",
    reason="Set ONNAX_TEST_MLX_EP=1 for native MLX GPU validation",
)
def test_medium_native_training_and_checkpoint(tmp_path):
    from examples.lora.mlx_training import run

    report = run(tmp_path, profile="medium", compiled_adam=True)
    assert report["same_parameter_acceptance"]
    assert report["independent_trajectory_acceptance"]
    assert report["strict_numerical_acceptance"]
    assert report["dimensions"] == {"input": 256, "hidden": 512, "output": 64, "rank": 16}
    assert report["parameter_bytes"] == (256 * 16 + 16 * 512) * 4
    assert report["forward_reload_batches"] == [1, 7, 128, 512]
    assert report["final_loss"] < report["initial_loss"] * 0.5
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
