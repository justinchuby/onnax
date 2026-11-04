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
