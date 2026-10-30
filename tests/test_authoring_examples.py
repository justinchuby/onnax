# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exercise actual ndonnx and ONNXScript exports through the same numerical task."""

import json

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples import ndonnx_autodiff, onnxscript_autodiff
from examples.authoring import run


@pytest.mark.parametrize(
    "frontend", [ndonnx_autodiff, onnxscript_autodiff], ids=["ndonnx", "onnxscript"]
)
def test_authored_graph_gradients_updates_and_roundtrip(frontend, tmp_path):
    proto, graph, gradient = frontend.differentiate()
    before = proto.SerializeToString()
    report = run((proto, graph, gradient), tmp_path)
    assert report["steps"] == 20
    assert report["final_loss"] < report["initial_loss"] * 0.01
    assert before == proto.SerializeToString()
    interface = json.loads((tmp_path / "interface.json").read_text())
    assert [p["input"] for p in interface["parameters"]] == ["weight", "bias"]
    assert set(gradient.graph.initializers).isdisjoint({"weight", "bias"})
    assert not graph.functions
    if frontend is onnxscript_autodiff:
        assert len(proto.functions) == 1
        assert all(not node.domain for node in graph.graph)


def test_frontends_agree_on_prediction_loss_and_gradients():
    feeds = {
        "features": np.array([[1, 2], [-1, 3], [0.5, -0.2]], np.float32),
        "target": np.array([1, -2, 0.3], np.float32),
        "weight": np.array([0.2, -0.4], np.float32),
        "bias": np.array(0.1, np.float32),
    }
    outputs = []
    for frontend in (ndonnx_autodiff, onnxscript_autodiff):
        _, _, gradient = frontend.differentiate()
        outputs.append(ReferenceEvaluator(ir.to_proto(gradient)).run(None, feeds))
    for a, b in zip(*outputs, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
