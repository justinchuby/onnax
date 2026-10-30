# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compiled external updates retain numerical, state and validation contracts."""

import os

import numpy as np
import pytest

from examples.lora.adam import AdamState
from examples.lora.mlx_adam import CompiledAdam
from examples.lora.mlx_exchange import synchronize
from examples.lora.mlx_training import host, run
from examples.lora.train import close, reference_update

native = pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1",
    reason="Set ONNAX_TEST_MLX_EP=1 for native MLX GPU validation",
)


@native
def test_compiled_updates_match_independent_moments_and_reuse_trace(monkeypatch):
    import mlx.core as mx

    import examples.lora.mlx_adam as module

    traces = []
    original = module._adam_arrays

    def observe(*args, **kwargs):
        traces.append(tuple(value.shape for value in args[0].values()))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_adam_arrays", observe)
    rng = np.random.default_rng(203)
    initial = {
        "lora.A": rng.normal(size=(4, 3)).astype(np.float32),
        "lora.B": rng.normal(size=(2, 5)).astype(np.float32),
    }
    expected = {name: value.copy() for name, value in initial.items()}
    with mx.stream(mx.gpu):
        update = CompiledAdam()
        params = {name: mx.array(value) for name, value in initial.items()}
        state = reference_state = None
        for _ in range(80):
            grads = {
                name: rng.normal(0, 0.1, size=value.shape).astype(np.float32)
                for name, value in expected.items()
            }
            inputs = {"p." + name: host(value) for name, value in params.items()}
            if state is not None:
                inputs.update(
                    {"m." + name: host(value) for name, value in state.first.items()}
                )
                inputs.update(
                    {"v." + name: host(value) for name, value in state.second.items()}
                )
            snapshots = {name: value.copy() for name, value in inputs.items()}
            params, state = update(
                params, {name: mx.array(value) for name, value in grads.items()}, state
            )
            synchronize(params, state)
            expected, reference_state = reference_update(expected, grads, reference_state)
            offset = 0
            for name, value in expected.items():
                close(host(params[name]), value)
                size = value.size
                close(
                    host(state.first[name]),
                    reference_state[1][offset : offset + size].reshape(value.shape),
                )
                close(
                    host(state.second[name]),
                    reference_state[2][offset : offset + size].reshape(value.shape),
                )
                offset += size
            for name, value in inputs.items():
                np.testing.assert_array_equal(value, snapshots[name])
        assert state.step == 80
        assert len(traces) == 1
        # A new shape is a new specialization, even with the same element count.
        params["lora.A"] = mx.reshape(params["lora.A"], (3, 4))
        changed = AdamState(
            state.step,
            {
                name: mx.reshape(value, params[name].shape)
                for name, value in state.first.items()
            },
            {
                name: mx.reshape(value, params[name].shape)
                for name, value in state.second.items()
            },
        )
        params, changed = update(
            params, {name: mx.ones_like(value) for name, value in params.items()}, changed
        )
        synchronize(params, changed)
        assert changed.step == 81
        assert len(traces) == 2
        expected["lora.A"] = expected["lora.A"].reshape(3, 4)
        expected, reference_state = reference_update(
            expected,
            {name: np.ones_like(value) for name, value in expected.items()},
            reference_state,
        )
        for name, value in expected.items():
            close(host(params[name]), value)


@native
def test_compiled_validation_runs_after_warmup():
    import mlx.core as mx

    with mx.stream(mx.gpu):
        params = {"p": mx.ones((2, 3), dtype=mx.float32)}
        grads = {"p": mx.ones((2, 3), dtype=mx.float32)}
        update = CompiledAdam()
        params, state = update(params, grads)
        synchronize(params, state)
        for bad_params, bad_grads, bad_state in [
            (params, {}, state),
            (params, {"p": mx.ones((3, 2), dtype=mx.float32)}, state),
            (
                params,
                grads,
                AdamState(
                    state.step, {"p": state.first["p"].astype(mx.float16)}, state.second
                ),
            ),
            (params, grads, AdamState(-1, state.first, state.second)),
            (params, grads, AdamState(True, state.first, state.second)),
        ]:
            with pytest.raises(ValueError):
                update(bad_params, bad_grads, bad_state)
        actual, state = update(params, grads, state)
        synchronize(actual, state)
        assert state.step == 2
        expected = np.full((2, 3), 0.94, np.float32)
        close(host(actual["p"]), expected)


@native
def test_compiled_configuration_uses_requested_hyperparameters():
    import mlx.core as mx

    with mx.stream(mx.gpu):
        update = CompiledAdam(lr=0.012, beta1=0.8, beta2=0.95, eps=1e-6)
        initial = np.array([1, -2, 0.5], np.float32)
        expected = initial.copy()
        m = np.zeros_like(expected)
        v = np.zeros_like(expected)
        params = {"p": mx.array(initial)}
        state = None
        for step in range(1, 8):
            gradient = np.array([0.2, -0.3, 0.4], np.float32) * np.float32((-1) ** step)
            m = 0.8 * m + 0.2 * gradient
            v = 0.95 * v + 0.05 * gradient * gradient
            expected = expected - 0.012 * (m / (1 - 0.8**step)) / (
                np.sqrt(v / (1 - 0.95**step)) + 1e-6
            )
            params, state = update(params, {"p": mx.array(gradient)}, state)
            synchronize(params, state)
            close(host(params["p"]), expected)


@native
def test_compiled_complete_training_and_fresh_checkpoint_updater(tmp_path):
    report = run(tmp_path, compiled_adam=True)
    assert report["compiled_adam"]
    assert report["steps"] == 200
    assert report["executed_providers"] == ["MLXExecutionProvider"]
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
