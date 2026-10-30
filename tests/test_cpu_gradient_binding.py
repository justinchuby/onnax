# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Reusable CPU loss/gradient outputs, input ownership and external update parity."""

import copy
import gc
import weakref

import array_api_compat
import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import optimizer

from examples.lora.accumulation import accumulate_gradients
from examples.lora.binding import CpuGradientBinding
from examples.lora.iobinding import run
from examples.lora.model import PARAMETERS, base_model, export
from examples.lora.train import close, problem, reference, session


@pytest.fixture
def binding_problem(tmp_path):
    w, c, x, y, _vx, _vy, params = problem()
    ir.save(base_model(w, c), tmp_path / "base.onnx")
    _, _, graph, interface = export(tmp_path / "base.onnx", tmp_path)
    ir.save(optimizer.optimize(graph.clone()), tmp_path / "gradient.optimized.onnx")
    executor = session(tmp_path / "gradient.optimized.onnx")
    return executor, interface, w, c, x, y, params


def test_complete_bound_training_and_measured_modes(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 200
    assert report["stable_output_addresses"]
    assert report["output_buffer_bytes"] == 644
    assert report["validation_loss"]["final"] < report["validation_loss"]["initial"] * 0.5
    for mode in ("run", "binding"):
        result = report["measurement"][mode]
        assert result["mode"] == mode
        assert result["peak_process_rss_bytes"] > 0
        for metric in result["timings_ms"].values():
            assert 0 <= metric["median"] <= metric["p95"]


def test_owners_rebinding_and_borrowed_output_lifetime(binding_problem):
    executor, interface, w, c, x, y, params = binding_problem
    bound = CpuGradientBinding(executor, interface)
    temporary = {n: v.copy() for n, v in params.items()}
    owners = {n: weakref.ref(v) for n, v in temporary.items()}
    bound.bind(x, y, temporary)
    del temporary
    gc.collect()
    assert all(owner() is not None for owner in owners.values())
    loss, first = bound.run()
    saved = {n: v.copy() for n, v in first.items()}
    expected = reference(x, y, params, w, c)
    close(loss, expected[1])
    for n in PARAMETERS:
        close(first[n], expected[2][n])
    feature_value = bound.input_values["features"]
    previous_parameter = bound.input_values[PARAMETERS[1]]
    changed = {n: p.copy() for n, p in params.items()}
    changed[PARAMETERS[1]].fill(0.1)
    bound.bind(x, y, changed)
    assert bound.input_values["features"] is feature_value
    assert bound.input_values[PARAMETERS[1]] is not previous_parameter
    loss, second = bound.run()
    expected = reference(x, y, changed, w, c)
    close(loss, expected[1])
    for n in PARAMETERS:
        assert first[n] is second[n]
        close(second[n], expected[2][n])
    # Shared input storage also makes in-place value changes visible to the runtime.
    changed[PARAMETERS[1]].fill(-0.1)
    loss, second = bound.run()
    expected = reference(x, y, changed, w, c)
    close(loss, expected[1])
    for n in PARAMETERS:
        close(second[n], expected[2][n])
    assert np.count_nonzero(saved[PARAMETERS[0]]) == 0
    assert np.linalg.norm(first[PARAMETERS[0]]) > 0
    for n, value in bound.input_values.items():
        assert value.data_ptr() == bound.input_arrays[n].ctypes.data
        assert np.shares_memory(value.numpy(), bound.input_arrays[n])
    for n, value in bound.output_values.items():
        assert value.data_ptr() == bound.output_arrays[n].ctypes.data
        assert np.shares_memory(value.numpy(), bound.output_arrays[n])


def test_invalid_feeds_invalidate_execution_and_overlap_is_rejected(binding_problem):
    executor, interface, _w, _c, x, y, params = binding_problem
    bound = CpuGradientBinding(executor, interface)
    with pytest.raises(RuntimeError, match="Bind inputs"):
        bound.run()
    overlapping = {
        **params,
        PARAMETERS[0]: bound.output_arrays[interface["parameters"][0]["gradient"]],
    }
    for xb, yb, supplied, message in (
        (x, y, {}, "exactly"),
        (x[:, ::-1], y, params, "contiguous"),
        (x.astype(np.float64), y, params, "FP32"),
        (x[:0], y[:0], params, "nonempty"),
        (x, y[:1], params, "matching"),
        (x, y, overlapping, "overlaps"),
    ):
        bound.bind(x, y, params)
        with pytest.raises(ValueError, match=message):
            bound.bind(xb, yb, supplied)
        with pytest.raises(RuntimeError, match="Bind inputs"):
            bound.run()


def test_binding_checks_runtime_interface_metadata(binding_problem):
    executor, interface, *_ = binding_problem
    for key, value in (("dtype", "float64"), ("gradient", "missing"), ("shape", [1, 1])):
        bad = copy.deepcopy(interface)
        bad["parameters"][0][key] = value
        with pytest.raises(ValueError):
            CpuGradientBinding(executor, bad)


def test_streaming_accumulation_consumes_borrowed_results_before_reuse(binding_problem):
    executor, interface, w, c, x, y, params = binding_problem
    params[PARAMETERS[1]].fill(0.05)
    bound = CpuGradientBinding(executor, interface)
    xp = array_api_compat.array_namespace(*params.values())

    def chunks():
        for start in range(0, len(x), 63):
            xb, yb = x[start : start + 63], y[start : start + 63]
            bound.bind(xb, yb, params)
            loss, grads = bound.run()
            yield len(xb), loss, grads

    loss, grads = accumulate_gradients(chunks(), xp=xp)
    expected = reference(x, y, params, w, c)
    close(loss, expected[1])
    for name in PARAMETERS:
        close(grads[name], expected[2][name])
        assert all(
            not np.shares_memory(grads[name], output)
            for output in bound.output_arrays.values()
        )
