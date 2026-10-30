# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Scaled classification graphs preserve explicit parameter and precision policies."""

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.mlp import NAMES, model
from examples.mlp_bf16 import problem as tiny_problem
from examples.mlp_bf16 import reference
from examples.mlp_device import PROFILES, models, problem, run


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_medium_classification_export_and_analytic_gradients(precision):
    x, y, vx, vy, p = problem("medium")
    train, forward, validation = models("medium", precision)
    for g in (train, forward, validation):
        onnx.checker.check_model(ir.to_proto(g), full_check=True)
        assert set(g.graph.initializers).isdisjoint(NAMES)
        assert not g.functions and all(not n.domain for n in g.graph)
    assert [v.name for v in train.graph.outputs] == [
        "loss",
        *[n + "_cotangent" for n in NAMES],
    ]
    assert sum(v.nbytes for v in p.values()) == 74256 * 4
    actual = ReferenceEvaluator(ir.to_proto(train)).run(
        None, {"features": x, "targets": y, **p}
    )
    expected = reference(x, y, p, precision=precision)
    for a, b in zip(actual, [expected[1], *expected[2].values()], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(y.sum(axis=-1), np.ones(len(x)))
    assert x.shape == (1024, 128) and vx.shape == (256, 128) and vy.shape == (256, 16)


def test_tiny_workload_preserves_existing_dataset_and_initialization():
    old, new = tiny_problem(), problem("tiny")
    for a, b in zip(old[:-1], new[:-1], strict=True):
        np.testing.assert_array_equal(a, b)
    for n in NAMES:
        np.testing.assert_array_equal(old[-1][n], new[-1][n])


def test_large_workload_has_four_explicit_parameter_inputs():
    x, y, vx, vy, p = problem("large")
    assert sum(v.size for v in p.values()) == 1116192
    assert x.shape == (2048, 512) and y.shape == (2048, 32)
    assert vx.shape == (512, 512) and vy.shape == (512, 32)
    g = model(input_width=512, hidden_width=2048, classes=32)
    assert [list(v.shape) for v in g.graph.inputs[2:]] == [list(p[n].shape) for n in NAMES]
    np.testing.assert_array_equal(y.sum(axis=-1), np.ones(len(y)))


@pytest.mark.parametrize(
    "kwargs", [{"input_width": 0}, {"hidden_width": True}, {"classes": -1}]
)
def test_invalid_model_dimensions(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        model(**kwargs)


def test_invalid_profile():
    with pytest.raises(ValueError, match="profile"):
        problem("missing")
    assert set(PROFILES) == {"tiny", "medium", "large"}


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
def test_medium_metal_master_state_training_and_recovery(tmp_path):
    report = run(tmp_path, profile="medium", mode="compiled", measure=False)
    assert report["same_parameter_acceptance"]
    assert report["shared_gradient_update_acceptance"]
    assert report["master_and_state_on_metal"] and report["pointer_identity_checked"]
    assert report["checkpoint_resume_verified"] and report["validation_accuracy"] >= 0.95
    assert report["executed_providers"] == ["MLXExecutionProvider"]
    assert report["parameter_bytes"] == 74256 * 4


def test_independent_update_honors_serialized_adam_configuration():
    from examples.lora.adam import adam_step
    from examples.mlp import reference_update

    params = problem("tiny")[-1]
    independent = {n: p.copy() for n, p in params.items()}
    state = reference_state = None
    rng = np.random.default_rng(90)
    options = dict(lr=0.01, eps=1e-6, beta1=0.8, beta2=0.95)
    for _ in range(5):
        grads = {n: rng.normal(0, 0.1, p.shape).astype(np.float32) for n, p in params.items()}
        params, state = adam_step(params, grads, state, xp=np, **options)
        independent, reference_state = reference_update(
            independent, grads, reference_state, **options
        )
        for n in NAMES:
            np.testing.assert_allclose(params[n], independent[n], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            np.concatenate([state.first[n].reshape(-1) for n in NAMES]),
            reference_state[1],
            rtol=1e-5,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.concatenate([state.second[n].reshape(-1) for n in NAMES]),
            reference_state[2],
            rtol=1e-5,
            atol=1e-12,
        )
