# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared CPU buffers, immutable gradient ownership and complete JAX update parity."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import optimizer

from examples.lora.accumulation import accumulate_gradients
from examples.lora.adam import adam_step
from examples.lora.dlpack import JaxCpuDLPack, synchronize
from examples.lora.jax_training import cpu, run
from examples.lora.model import PARAMETERS, base_model, export
from examples.lora.train import close, problem, reference, reference_update, session


@pytest.fixture
def dlpack_problem(tmp_path):
    w, c, x, y, _vx, _vy, params = problem()
    ir.save(base_model(w, c), tmp_path / "base.onnx")
    _, _, graph, interface = export(tmp_path / "base.onnx", tmp_path)
    ir.save(optimizer.optimize(graph.clone()), tmp_path / "gradient.optimized.onnx")
    return session(tmp_path / "gradient.optimized.onnx"), interface, w, c, x, y, params


def test_complete_jax_dlpack_training_checkpoint_and_measured_modes(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 200
    assert report["checkpoint_resume_verified"]
    assert report["dlpack_shared_input_output_pointers"]
    assert report["validation_loss"]["final"] < report["validation_loss"]["initial"] * 0.5
    for mode in ("dlpack", "numpy"):
        measurement = report["measurement"][mode]
        assert measurement["mode"] == mode
        assert measurement["peak_process_rss_bytes"] > 0
        for metric in measurement["timings_ms"].values():
            assert 0 <= metric["median"] <= metric["p95"]


def test_dlpack_input_owners_and_retained_outputs_survive_reexecution(dlpack_problem):
    executor, interface, w, c, x, y, params = dlpack_problem
    bound = JaxCpuDLPack(executor, interface)
    jx, jy = cpu((x, y))
    temporary = cpu(params)
    refs = {n: weakref.ref(p) for n, p in temporary.items()}
    bound.bind(jx, jy, temporary)
    del temporary
    gc.collect()
    assert all(ref() is not None for ref in refs.values())
    first_loss, first = bound.run()
    saved_loss = np.asarray(first_loss).copy()
    saved = {n: np.asarray(value).copy() for n, value in first.items()}
    pointers = {n: value.data_ptr() for n, value in bound.output_values.items()}
    for n, value in bound.input_values.items():
        assert value.data_ptr() == bound.input_arrays[n].unsafe_buffer_pointer()
    feature_handle = bound.input_values["features"]
    parameter_handle = bound.input_values[PARAMETERS[1]]
    changed = {n: p.copy() for n, p in params.items()}
    changed[PARAMETERS[1]].fill(0.1)
    bound.bind(jx, jy, cpu(changed))
    assert bound.input_values["features"] is feature_handle
    assert bound.input_values[PARAMETERS[1]] is not parameter_handle
    second_loss, second = bound.run()
    expected = reference(x, y, changed, w, c)
    close(np.asarray(second_loss), expected[1])
    for n in PARAMETERS:
        close(np.asarray(second[n]), expected[2][n])
        assert (
            second[n].unsafe_buffer_pointer()
            == bound.output_values[bound.gradient_names[n]].data_ptr()
        )
    assert all(value.data_ptr() != pointers[n] for n, value in bound.output_values.items())
    del bound
    gc.collect()
    np.testing.assert_array_equal(np.asarray(first_loss), saved_loss)
    for n in PARAMETERS:
        np.testing.assert_array_equal(np.asarray(first[n]), saved[n])
    assert np.linalg.norm(np.asarray(second[PARAMETERS[0]])) > 0


def test_pending_adam_consumption_keeps_original_gradients(dlpack_problem):
    executor, interface, _w, _c, x, y, params = dlpack_problem
    params[PARAMETERS[1]].fill(0.05)
    jx, jy, initial = cpu((x, y, params))
    bound = JaxCpuDLPack(executor, interface)
    bound.bind(jx, jy, initial)
    _, grads = bound.run()
    saved = {n: np.asarray(g).copy() for n, g in grads.items()}
    updated, state = adam_step(initial, grads, xp=jnp)
    # Another native execution happens before the explicit optimizer boundary wait.
    bound.execute()
    synchronize(updated, state)
    expected, _ = reference_update(params, saved, None)
    for n in PARAMETERS:
        np.testing.assert_array_equal(np.asarray(grads[n]), saved[n])
        close(np.asarray(updated[n]), expected[n])


def test_invalid_jax_feeds_invalidate_execution(dlpack_problem):
    executor, interface, _w, _c, x, y, params = dlpack_problem
    bound = JaxCpuDLPack(executor, interface)
    with pytest.raises(RuntimeError, match="Bind inputs"):
        bound.execute()
    with pytest.raises(RuntimeError, match="Execute"):
        bound.import_outputs()
    jx, jy, jp = cpu((x, y, params))
    for a, b, p in (
        (x, jy, jp),
        (jx.astype(jnp.float16), jy, jp),
        (jx[:0], jy[:0], jp),
        (jx, jy[:1], jp),
        (jx, jy, {}),
    ):
        bound.bind(jx, jy, jp)
        with pytest.raises(ValueError):
            bound.bind(a, b, p)
        with pytest.raises(RuntimeError, match="Bind inputs"):
            bound.execute()
    assert jax.devices("cpu")


def test_imported_gradient_storage_outlives_its_session(dlpack_problem, tmp_path):
    _executor, interface, w, c, x, y, params = dlpack_problem
    params[PARAMETERS[1]].fill(0.05)
    expected = reference(x, y, params, w, c)

    def produce():
        local_session = session(tmp_path / "gradient.optimized.onnx")
        local = JaxCpuDLPack(local_session, interface)
        local.bind(*cpu((x, y)), cpu(params))
        return local.run()

    loss, gradients = produce()
    gc.collect()
    close(np.asarray(loss), expected[1])
    for n in PARAMETERS:
        close(np.asarray(gradients[n]), expected[2][n])


def test_owned_dlpack_gradients_feed_streaming_jax_accumulation(dlpack_problem):
    executor, interface, w, c, x, y, params = dlpack_problem
    params[PARAMETERS[1]].fill(0.05)
    jx, jy, jp = cpu((x, y, params))
    bound = JaxCpuDLPack(executor, interface)

    def chunks():
        for start in range(0, len(x), 63):
            xb, yb = jx[start : start + 63], jy[start : start + 63]
            bound.bind(xb, yb, jp)
            loss, grads = bound.run()
            yield len(xb), loss, grads

    loss, grads = accumulate_gradients(chunks(), xp=jnp)
    jax.block_until_ready([loss, *grads.values()])
    expected = reference(x, y, params, w, c)
    close(np.asarray(loss), expected[1])
    for n in PARAMETERS:
        close(np.asarray(grads[n]), expected[2][n])
