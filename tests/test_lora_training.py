# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Real external optimization, frozen-weight traversal and serialized interfaces."""

from __future__ import annotations

import copy
import sys

import numpy as np
import onnx_ir as ir
import pytest

from examples.lora.adam import adam_step
from examples.lora.model import PARAMETERS, adapt, base_model, export, validate_interface
from examples.lora.train import evaluate, problem, reference_update, run, session


def test_complete_lora_training_and_roundtrip(tmp_path):
    report = run(tmp_path, measure=True)
    assert report["steps"] == 200
    measurement = report["measurement"]
    assert measurement["peak_process_rss_bytes"] > 0
    assert measurement["model_read_ms"] >= 0
    for metric in measurement["timings_ms"].values():
        assert 0 <= metric["median"] <= metric["p95"]
    for name in ("train_loss", "validation_loss"):
        assert report[name]["final"] < 0.5 * report[name]["initial"]
    assert report["parameter_bytes"] == (16 * 4 + 4 * 24) * 4


def test_parameter_collisions_and_invalid_interfaces(tmp_path):
    w, c, *_ = problem()
    base = base_model(w, c)
    before = ir.to_proto(base).SerializeToString()
    base.graph.inputs[0].name = "lora.A"
    with pytest.raises(ValueError, match="collision"):
        adapt(base)
    base = ir.from_proto(ir.to_proto(base_model(w, c)))
    ir.save(base, tmp_path / "base.onnx")
    _, _, graph, interface = export(tmp_path / "base.onnx", tmp_path)
    assert before == ir.to_proto(base).SerializeToString()
    for field, value in (
        ("input", "missing"),
        ("gradient", "missing"),
        ("shape", [1]),
        ("dtype", "float64"),
    ):
        bad = copy.deepcopy(interface)
        bad["parameters"][0][field] = value
        with pytest.raises(ValueError):
            validate_interface(graph, bad)
    bad = copy.deepcopy(interface)
    bad["frozen"]["base.W"] = "changed"
    with pytest.raises(ValueError, match="Frozen weight mismatch"):
        validate_interface(graph, bad)


@pytest.mark.parametrize("backend", ["numpy", "strict", "jax", "mlx", "cupy"])
def test_array_api_adam_matches_independent_numpy_updates(backend):
    import array_api_compat

    if backend == "numpy":
        xp = array_api_compat.array_namespace(np.zeros(1, np.float32))
    elif backend == "strict":
        xp = pytest.importorskip("array_api_strict")
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_platform_name", "cpu")
        xp = jax.numpy
    elif backend == "mlx":
        if sys.platform != "darwin":
            pytest.skip("MLX test requires Apple Silicon; not validated here")
        try:
            import mlx.core as mx
        except ImportError as error:
            pytest.skip(f"MLX runtime unavailable: {error}")
        mx.set_default_device(mx.gpu)
        xp = mx.array([0.0]).__array_namespace__()
    else:
        try:
            import cupy as cp
        except ImportError as error:
            pytest.skip(f"CuPy unavailable: {error}")
        try:
            devices = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as error:
            pytest.skip(f"CUDA runtime unavailable: {error}")
        if devices == 0:
            pytest.skip("CuPy test requires CUDA; not validated here")
        xp = array_api_compat.array_namespace(cp.zeros(1, dtype=cp.float32))
    params = {
        PARAMETERS[0]: xp.asarray([[0.2, -0.3]], dtype=xp.float32),
        PARAMETERS[1]: xp.asarray([[0.1], [-0.4]], dtype=xp.float32),
    }
    expected = {
        PARAMETERS[0]: np.array([[0.2, -0.3]], np.float32),
        PARAMETERS[1]: np.array([[0.1], [-0.4]], np.float32),
    }
    state = reference_state = None
    for step in range(20):
        grads = {
            name: xp.full_like(value, 0.01 * (step + 1)) for name, value in params.items()
        }
        expected_grads = {
            name: np.full_like(value, 0.01 * (step + 1)) for name, value in expected.items()
        }
        old = params
        params, state = adam_step(params, grads, state, xp=xp)
        expected, reference_state = reference_update(expected, expected_grads, reference_state)
        for name in PARAMETERS:
            # Transfers/synchronization are explicit test boundaries, not optimizer work.
            if backend == "cupy":
                actual = cp.asnumpy(params[name])
            else:
                actual = np.asarray(params[name])
            np.testing.assert_allclose(actual, expected[name], rtol=1e-4, atol=1e-6)
            assert params[name] is not old[name]
            assert params[name].dtype == xp.float32
            assert state.first[name].dtype == xp.float32
            assert state.second[name].dtype == xp.float32
    with pytest.raises(ValueError, match="names must match"):
        adam_step(params, {}, state, xp=xp)


def test_gradient_output_collision_uses_exported_mapping(tmp_path):
    w, c, *_ = problem()
    base = base_model(w, c)
    next(iter(base.graph)).outputs[0].name = "lora.A_cotangent"
    ir.save(base, tmp_path / "base.onnx")
    _, _, graph, interface = export(tmp_path / "base.onnx", tmp_path)
    validate_interface(graph, interface)
    assert interface["parameters"][0]["gradient"] != "lora.A_cotangent"
    loaded = ir.load(tmp_path / "gradient.onnx")
    validate_interface(loaded, interface)


def test_array_api_adam_rejects_mismatched_gradient_and_state():
    import array_api_compat

    from examples.lora.adam import AdamState

    xp = array_api_compat.array_namespace(np.zeros(1, np.float32))
    params = {"p": np.ones(2, np.float32)}
    with pytest.raises(ValueError, match="Gradient shape/dtype mismatch"):
        adam_step(params, {"p": np.ones(3, np.float32)}, xp=xp)
    bad = AdamState(0, {"p": np.zeros(3, np.float32)}, {"p": np.zeros(2, np.float32)})
    with pytest.raises(ValueError, match="Adam state shape/dtype mismatch"):
        adam_step(params, params, bad, xp=xp)


def test_lora_runtime_rejects_missing_parameters_and_broadcast_targets(tmp_path):
    w, c, x, y, _, _, params = problem()
    ir.save(base_model(w, c), tmp_path / "base.onnx")
    _, _, _, interface = export(tmp_path / "base.onnx", tmp_path)
    executor = session(tmp_path / "gradient.onnx")
    with pytest.raises(ValueError, match="Runtime feed mismatch"):
        evaluate(executor, interface, x, y, {PARAMETERS[0]: params[PARAMETERS[0]]})
    with pytest.raises(ValueError, match="target"):
        evaluate(executor, interface, x, y[:1], params)
