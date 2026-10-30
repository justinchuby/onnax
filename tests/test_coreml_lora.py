# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable numerical checks for the example's CoreML representation."""

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

import onnax
from examples.lora.coreml import gradient_model
from examples.lora.train import close, problem, reference


@pytest.mark.parametrize("scale", [0, 1, 20])
def test_coreml_representation_preserves_gradients_and_hvp(scale):
    w, c, x, y, _, _, params = problem()
    x, y = x[:3] * np.float32(max(scale, 1)), y[:3]
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    if scale == 0:
        w = np.zeros_like(w)
        params["lora.B"] = np.zeros_like(params["lora.B"])
    feeds = {"features": x, "target": y, **params}
    graphs = [gradient_model(w, c, batch=3, relu_magnitude=mode) for mode in (False, True)]
    expected = reference(x, y, params, w, c)
    results = []
    for graph in graphs:
        onnx.checker.check_model(ir.to_proto(graph), full_check=True)
        actual = ReferenceEvaluator(ir.to_proto(graph)).run(None, feeds)
        for a, b in zip(actual, [*expected[:2], *expected[2].values()], strict=True):
            close(a, b)
        hvp = onnax.jvp(graph, inputs=["lora.A", "lora.B"])
        seeds = {
            hvp.graph.inputs[-2].name: np.full_like(params["lora.A"], 0.2),
            hvp.graph.inputs[-1].name: np.full_like(params["lora.B"], -0.1),
        }
        results.append(ReferenceEvaluator(ir.to_proto(hvp)).run(None, {**feeds, **seeds}))
    if scale == 0:
        assert any(np.count_nonzero(value) for value in results[0][-2:])
    for a, b in zip(*results, strict=True):
        close(a, b)
    assert all(
        node.op_type not in {"Where", "Neg", "GreaterOrEqual"} for node in graphs[1].graph
    )


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_static_batch(batch):
    w, c, *_ = problem()
    with pytest.raises(ValueError, match="positive integer"):
        gradient_model(w, c, batch=batch)


def test_static_graph_cpu_execution():
    from examples.lora.train import session

    w, c, x, y, _, _, params = problem()
    graph = gradient_model(w, c, batch=256)
    executor = session(ir.to_proto(graph).SerializeToString())
    actual = executor.run(None, {"features": x, "target": y, **params})
    expected = reference(x, y, params, w, c)
    for a, b in zip(actual, [*expected[:2], *expected[2].values()], strict=True):
        close(a, b)


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_COREML") != "1",
    reason="Set ONNAX_TEST_COREML=1 on a CoreML-enabled Mac for native training",
)
def test_native_coreml_training(tmp_path):
    from examples.lora.coreml import train

    report = train(tmp_path)
    assert report["executed_providers"] == ["CoreMLExecutionProvider"]
    assert report["steps"] == 200


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_COREML") != "1",
    reason="Set ONNAX_TEST_COREML=1 for native large-batch validation",
)
def test_native_large_batch_nonzero_adapters():
    from examples.lora.coreml import session

    w, c, x, y, _, _, params = problem()
    x, y = np.tile(x, (64, 1)), np.tile(y, (64, 1))
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph = gradient_model(w, c, batch=len(x))
    actual = session(graph).run(None, {"features": x, "target": y, **params})
    expected = reference(x, y, params, w, c)
    for a, b in zip(actual, [*expected[:2], *expected[2].values()], strict=True):
        close(a, b)
