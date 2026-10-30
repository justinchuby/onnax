# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Complete classifier training and numerically stable extreme-logit losses."""

import numpy as np
import onnx_ir as ir

import onnax
from examples.lora.train import close, session
from examples.mlp import NAMES, SHAPES, model, reference, run


def test_complete_mlp_training(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 150
    assert report["parameter_bytes"] == 99 * 4
    assert report["measurement"]["peak_process_rss_bytes"] > 0
    for metric in report["measurement"]["timings_ms"].values():
        assert 0 <= metric["median"] <= metric["p95"]
    assert report["final_train_loss"] < report["initial_train_loss"] * 0.5
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
    assert report["validation_accuracy"] >= 0.95


def test_classifier_loss_is_stable_and_shift_invariant(tmp_path):
    graph = onnax.grad(model(), inputs=list(NAMES), output="loss")
    ir.save(graph, tmp_path / "gradient.onnx")
    executor = session(tmp_path / "gradient.onnx")
    params = {n: np.zeros(s, np.float32) for n, s in zip(NAMES, SHAPES, strict=True)}
    params[NAMES[-1]] = np.array([-1000, 0, 1000], np.float32)
    feeds = {
        "features": np.zeros((2, 2), np.float32),
        "targets": np.array([[1, 0, 0], [1, 0, 0]], np.float32),
    }
    first = executor.run(None, {**feeds, **params})
    params[NAMES[-1]] += 10000
    second = executor.run(None, {**feeds, **params})
    for result in (first, second):
        assert all(np.isfinite(value).all() for value in result)
        close(result[1], np.array(2000, np.float32))
        close(result[-1], np.array([-1, 0, 1], np.float32))
    for a, b in zip(first[1:], second[1:], strict=True):
        close(a, b)


def test_mlp_reference_gradients_against_central_differences():
    rng = np.random.default_rng(12)
    params = {n: rng.normal(0, 0.2, s) for n, s in zip(NAMES, SHAPES, strict=True)}
    x = rng.normal(size=(5, 2))
    target = np.eye(3)[[0, 1, 2, 1, 0]]
    gradients = reference(x, target, params)[2]
    for n in NAMES:
        coordinate = (0,) * params[n].ndim
        plus = {k: v.copy() for k, v in params.items()}
        minus = {k: v.copy() for k, v in params.items()}
        plus[n][coordinate] += 1e-5
        minus[n][coordinate] -= 1e-5
        difference = (reference(x, target, plus)[1] - reference(x, target, minus)[1]) / 2e-5
        close(difference, gradients[n][coordinate])
