# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Weighted FP32 streaming gradients, external updates and runtime boundaries."""

import json

import array_api_compat
import numpy as np
import pytest

from examples.lora.accumulation import accumulate_gradients
from examples.lora.adam import adam_step
from examples.lora.microbatch import run
from examples.lora.model import PARAMETERS
from examples.lora.train import close, reference_update


def test_complete_microbatch_training(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 200
    assert report["microbatch_sizes"] == [63, 63, 63, 63, 4]
    assert report["loss_history"][-1] < report["loss_history"][0] * 0.5
    validation = report["validation_loss"]
    assert validation["final"] < validation["initial"] * 0.5
    assert set(report["dtype_policy"].values()) == {"float32"}
    assert report["bf16_probe"]["checker"] == "passed"
    assert report["bf16_probe"]["autodiff"]["supported"] is False
    for size, measured in report["measurement"].items():
        assert measured["microbatch"] == int(size)
        assert measured["effective_batch"] == 16384
        assert 0 < measured["median_ms"] <= measured["p95_ms"]
        assert measured["peak_process_rss_bytes"] > 0
    assert json.loads((tmp_path / "report.json").read_text())["steps"] == 200


@pytest.mark.parametrize("backend", ["numpy", "strict", "jax"])
def test_array_api_accumulation_and_multistep_adam(backend):
    if backend == "numpy":
        xp = array_api_compat.array_namespace(np.zeros(1, np.float32))
    elif backend == "strict":
        import array_api_strict

        xp = array_api_strict
    else:
        import jax

        jax.config.update("jax_platform_name", "cpu")
        xp = jax.numpy
    params = {name: xp.asarray([[0.2, -0.3]], dtype=xp.float32) for name in PARAMETERS}
    independent = {name: np.array([[0.2, -0.3]], np.float32) for name in PARAMETERS}
    state = ref_state = None
    for step in range(20):
        chunks = [
            (
                count,
                xp.asarray(float(i), dtype=xp.float32),
                {n: xp.full_like(p, (step + 1) * i / 100) for n, p in params.items()},
            )
            for i, count in enumerate((3, 2, 1), 1)
        ]
        loss, grads = accumulate_gradients(iter(chunks), xp=xp)
        close(np.asarray(loss), np.float32(10 / 6))
        for name in PARAMETERS:
            close(np.asarray(grads[name]), np.full((1, 2), (step + 1) / 60, np.float32))
            assert grads[name].dtype == xp.float32
        # Boundary conversions are test-only; accumulation and Adam never read array scalars.
        for i, (_, _, chunk) in enumerate(chunks, 1):
            close(
                np.asarray(chunk[PARAMETERS[0]]),
                np.full((1, 2), (step + 1) * i / 100, np.float32),
            )
        params, state = adam_step(params, grads, state, xp=xp)
        expected_grads = {n: np.full((1, 2), (step + 1) / 60, np.float32) for n in PARAMETERS}
        independent, ref_state = reference_update(independent, expected_grads, ref_state)
        assert state.step == step + 1
        for name in PARAMETERS:
            close(np.asarray(params[name]), independent[name])


def test_accumulation_rejects_invalid_microbatches():
    xp = array_api_compat.array_namespace(np.zeros(1, np.float32))
    loss = np.array(1, np.float32)
    good = {"p": np.ones(2, np.float32)}
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            accumulate_gradients([(count, loss, good)], xp=xp)
    with pytest.raises(ValueError, match="At least one"):
        accumulate_gradients([], xp=xp)
    for bad_loss, bad_grads in (
        (np.ones(1, np.float32), good),
        (np.array(1, np.float64), good),
        (loss, {}),
    ):
        with pytest.raises(ValueError, match="scalar FP32"):
            accumulate_gradients([(1, bad_loss, bad_grads)], xp=xp)
    for bad_grads in (
        {"q": good["p"]},
        {"p": np.ones(3, np.float32)},
        {"p": np.ones(2, np.float64)},
    ):
        with pytest.raises(ValueError, match=r"names|shape/dtype"):
            accumulate_gradients([(3, loss, good), (1, loss, bad_grads)], xp=xp)
