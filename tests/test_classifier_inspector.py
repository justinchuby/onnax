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
    assert all(v["mismatched"] == 0 for v in result["observed_primal_head_gradients"].values())
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
