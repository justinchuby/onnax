# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Check observed-primal classifier diagnostics independently of Metal."""

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from examples.classifier_inspector import head_reference, inspect, instrument, replay
from examples.mlp import NAMES
from examples.mlp_bf16 import graph, problem


def test_observed_primal_gradients_and_preservation():
    x, y, _, _, params = problem()
    model = graph(batch=3)
    feeds = {"features": x[:3], "targets": y[:3], **params}
    import onnx_ir as ir

    original = ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)
    observed, count = instrument(model)
    evaluator = ReferenceEvaluator(ir.to_proto(observed))

    class Executor:
        def run(self, names, data):
            return evaluator.run(names, data)

        def get_outputs(self):
            return observed.graph.outputs

    mapping = dict(zip(NAMES, [v.name for v in model.graph.outputs[2:]], strict=True))
    result = inspect(Executor(), feeds, original, mapping, original_count=count)
    assert result["original_outputs_equal"]
    assert all(v["mismatched"] == 0 for v in result["observed_primal_gradients"].values())
    changed = [v.copy() for v in original]
    changed[0].flat[0] += 1
    with pytest.raises(ValueError, match="changed original"):
        inspect(Executor(), feeds, changed, mapping, original_count=count)


def test_head_reference_matches_finite_difference():
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(5, 4))
    weight = rng.normal(size=(4, 3))
    bias = rng.normal(size=3)
    targets = np.eye(3)[[0, 1, 2, 0, 2]]
    gradients = head_reference(hidden, hidden @ weight + bias, targets)

    def loss(w):
        scores = hidden @ w + bias
        shifted = scores - scores.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -np.mean(np.sum(targets * logp, axis=1))

    for index in np.ndindex(weight.shape):
        plus, minus = weight.copy(), weight.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        np.testing.assert_allclose(
            gradients["layer2.weight"][index],
            (loss(plus) - loss(minus)) / 2e-5,
            rtol=1e-7,
            atol=1e-9,
        )


@pytest.mark.parametrize("step", [-1, True, 0.5])
def test_replay_rejects_invalid_step(step):
    with pytest.raises(ValueError, match="nonnegative integer"):
        replay("unused", step=step)


def test_full_observed_backward_matches_smooth_surrogate_finite_difference():
    from examples.classifier_inspector import reference

    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 2))
    params = {
        "layer1.weight": rng.normal(size=(2, 3)),
        "layer1.bias": rng.normal(size=3),
        "layer2.weight": rng.normal(size=(3, 2)),
        "layer2.bias": rng.normal(size=2),
    }
    targets = np.eye(2)[[0, 1, 1, 0]]

    def forward(p):
        z = x @ p["layer1.weight"] + p["layer1.bias"]
        hidden = np.tanh(z)
        logits = hidden @ p["layer2.weight"] + p["layer2.bias"]
        return z, hidden, logits

    def loss(p):
        logits = forward(p)[2]
        shifted = logits - logits.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -np.mean(np.sum(targets * logp, axis=1))

    gradients = reference(x, params["layer2.weight"], *forward(params), targets)
    for name, value in params.items():
        for index in np.ndindex(value.shape):
            plus, minus = (
                {n: v.copy() for n, v in params.items()},
                {n: v.copy() for n, v in params.items()},
            )
            plus[name][index] += 1e-5
            minus[name][index] -= 1e-5
            np.testing.assert_allclose(
                gradients[name][index], (loss(plus) - loss(minus)) / 2e-5, rtol=1e-7, atol=1e-9
            )


def test_audit_replays_saved_graphs_without_metal(tmp_path, monkeypatch):
    import json

    import onnx_ir as ir

    from examples.classifier_inspector import audit
    from examples.lora import mlx_runtime

    x, y, _, _, params = problem()
    model = graph(batch=3)
    ir.save(model, tmp_path / "gradient.onnx")
    np.savez(tmp_path / "benchmark.npz", features=x[:3], targets=y[:3], **params)
    (tmp_path / "interface.json").write_text(
        json.dumps(
            {
                "profile": "tiny",
                "adam": {"lr": 0.03},
                "parameters": [
                    {"input": n, "gradient": v.name}
                    for n, v in zip(NAMES, model.graph.outputs[2:], strict=True)
                ],
            }
        )
    )

    class Executor:
        def __init__(self, g):
            self.model = g
            self.evaluator = ReferenceEvaluator(ir.to_proto(g))

        def run(self, outputs, feeds):
            return self.evaluator.run(outputs, feeds)

        def get_outputs(self):
            return self.model.graph.outputs

    monkeypatch.setattr(mlx_runtime, "session", Executor)
    result = audit(tmp_path, steps=3)
    assert result["steps"] == 3 and result["original_outputs_equal"]
    assert len(result["comparisons"]) == 8
    assert all(v["failed_steps"] == 0 for v in result["comparisons"].values())
    for count in (0, -1, True, 0.5):
        with pytest.raises(ValueError, match="positive integer"):
            audit(tmp_path, steps=count)


def test_instrument_rejects_other_model_structures():
    with pytest.raises(ValueError, match="two BF16"):
        instrument(graph(precision="fp32", batch=3))
    model = graph(batch=3)
    for node in model.graph:
        for value in node.outputs:
            if value.name == "logits":
                value.name = "renamed_scores"
    with pytest.raises(ValueError, match="named logits"):
        instrument(model)


def test_native_audit_when_enabled(tmp_path):
    import json
    import os

    import onnx_ir as ir

    from examples.classifier_inspector import audit
    from examples.mlp_device import models
    from examples.mlp_device import problem as scaled_problem

    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    x, y, _, _, params = scaled_problem("medium")
    train, _, _ = models("medium", "bf16")
    ir.save(train, tmp_path / "gradient.onnx")
    np.savez(tmp_path / "benchmark.npz", features=x, targets=y, **params)
    (tmp_path / "interface.json").write_text(
        json.dumps(
            {
                "profile": "medium",
                "adam": {"lr": 0.01, "eps": 1e-6},
                "parameters": [
                    {"input": n, "gradient": v.name}
                    for n, v in zip(NAMES, train.graph.outputs[1:], strict=True)
                ],
            }
        )
    )
    report = audit(tmp_path, steps=150)
    assert report["original_outputs_equal"] and report["steps"] == 150
    observed = {
        n: v for n, v in report["comparisons"].items() if n.startswith("observed_primal")
    }
    assert len(observed) == 4
    assert all(v["failed_steps"] == 0 for v in observed.values())
