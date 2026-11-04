# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute matrix-free FP32 adapter products against independent references."""

import json

import numpy as np
import onnx_ir as ir
import pytest

pytest.importorskip("torch")

from examples.lora.curvature import dot, interface, run
from examples.lora.model import PARAMETERS
from examples.lora.train import problem, session


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("curvature") / "models"
    report = run(directory)
    return directory, report


def test_lora_curvature_references_and_initial_negative_curvature(artifacts):
    directory, report = artifacts
    assert report == json.loads((directory / "report.json").read_text())
    assert report["dtype"] == "float32"
    assert report["initial_a_gradient_zero"]
    assert report["negative_direction_curvature"] < 0
    assert report["gauss_newton_direction_curvature"] > 0
    assert len(list(directory.glob("*.onnx"))) == 6


def test_external_gauss_newton_adapter_updates_and_reload(artifacts):
    directory, report = artifacts
    fitting = report["fit"]
    steps = fitting["steps"]
    assert 0 < len(steps) <= 12
    assert fitting["train_loss"][1] < fitting["train_loss"][0] * 0.5
    assert fitting["validation_loss"][1] < fitting["validation_loss"][0] * 0.5
    assert steps[0]["loss"] == fitting["train_loss"][0]
    assert steps[-1]["next_loss"] == fitting["train_loss"][1]
    assert all(row["next_loss"] < row["loss"] for row in steps)
    assert all(0 < row["step_size"] <= 1 for row in steps)
    assert all(row["linear_residual"] <= 2 * row["linear_tolerance"] for row in steps)
    assert fitting["reference_products"] > len(steps)
    assert fitting["product_max_abs_error"] < 1e-6
    assert fitting["parameter_inputs"] == list(PARAMETERS)
    with np.load(directory / fitting["adapter"], allow_pickle=False) as archive:
        params = {name: archive[name] for name in PARAMETERS}
    assert all(value.dtype == np.float32 for value in params.values())
    runtime = session(directory / "hvp.optimized.onnx")
    model = ir.load(directory / "hvp.optimized.onnx")
    _, _, x, target, _, _, initial = problem()
    feeds = {"features": x, "target": target, **params}
    feeds.update(
        {
            seed.name: np.zeros_like(initial[name])
            for seed, name in zip(model.graph.inputs[-2:], PARAMETERS, strict=True)
        }
    )
    (loss,) = runtime.run(["loss"], feeds)
    np.testing.assert_array_equal(loss, np.float32(fitting["train_loss"][1]))


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_invalid_fit_step_count(tmp_path, steps):
    with pytest.raises(ValueError, match="nonnegative integer"):
        run(tmp_path / "invalid", fit_steps=steps)


@pytest.mark.parametrize("suffix", ["", ".optimized"])
def test_exported_hvp_has_symmetric_parameter_blocks(artifacts, suffix):
    directory, _ = artifacts
    path = directory / f"hvp{suffix}.onnx"
    model = ir.load(path)
    runtime = session(path)
    _, _, x, target, _, _, params = problem()
    rng = np.random.default_rng(20261103)
    params[PARAMETERS[1]] = rng.normal(0, 0.02, params[PARAMETERS[1]].shape).astype(np.float32)
    feeds = {"features": x[:5], "target": target[:5], **params}
    u, v = (
        [rng.normal(size=params[name].shape).astype(np.float32) for name in PARAMETERS]
        for _ in range(2)
    )

    def product(direction):
        return runtime.run(
            interface(model)["derivative_outputs"],
            {
                **feeds,
                **{
                    seed.name: value
                    for seed, value in zip(model.graph.inputs[-2:], direction, strict=True)
                },
            },
        )

    np.testing.assert_allclose(dot(u, product(v)), dot(v, product(u)), rtol=1e-4, atol=1e-6)
