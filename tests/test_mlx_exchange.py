# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared views, native ownership/rebinding and complete MLX GPU training."""

import gc
import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora.mlx_exchange import CpuView, MlxDLPack
from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import gradient_model, host, run
from examples.lora.train import close, problem, reference

native = pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1",
    reason="Set ONNAX_TEST_MLX_EP=1 on an MLX-enabled Mac",
)


def test_explicit_cpu_dlpack_view_retains_owner_and_rejects_copy_requests():
    array = np.arange(8, dtype=np.float32)

    class Producer:
        # Keep this portable protocol test compatible with core NumPy 1.26.
        # The real Metal producer is exercised by the opt-in native tests.
        def __init__(self, owner):
            self.owner = owner

        def __dlpack__(self, **kwargs):
            assert kwargs["dl_device"] == (1, 0)
            assert kwargs["copy"] is False
            return self.owner.__dlpack__()

    view = CpuView(Producer(array))
    value = ort.OrtValue.from_dlpack(view)
    assert value.data_ptr() == array.ctypes.data
    del view, array
    gc.collect()
    np.testing.assert_array_equal(value.numpy(), np.arange(8, dtype=np.float32))
    view = CpuView(np.ones(3, np.float32))
    for kwargs in [{"copy": True}, {"dl_device": (8, 0)}]:
        with pytest.raises(BufferError, match="shared CPU"):
            view.__dlpack__(**kwargs)


def test_fp32_native_gradient_export_is_plain_and_matches_reference():
    w, c, x, y, _, _, params = problem()
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph, interface = gradient_model(w, c, batch=3)
    onnx.checker.check_model(ir.to_proto(graph), full_check=True)
    assert len(graph.graph.outputs) == 3
    assert interface["prediction"] is None
    actual = ReferenceEvaluator(ir.to_proto(graph)).run(
        None, {"features": x[:3], "target": y[:3], **params}
    )
    expected = reference(x[:3], y[:3], params, w, c)
    for a, b in zip(actual, [expected[1], *expected[2].values()], strict=True):
        close(a, b)
    assert [row["input"] for row in interface["parameters"]] == ["lora.A", "lora.B"]
    assert not any(
        node.domain or node.op_type in {"NonZero", "Reshape"} for node in graph.graph
    )


@pytest.fixture
def native_problem():
    import mlx.core as mx

    with mx.stream(mx.gpu):
        w, c, x, y, _, _, params = problem()
        params["lora.B"] = np.full_like(params["lora.B"], 0.05)
        graph, interface = gradient_model(w, c, batch=7, include_prediction=True)
        executor = session(graph)
        bound = MlxDLPack(executor, interface)
        arrays = {name: mx.array(value) for name, value in params.items()}
        yield mx, bound, mx.array(x[:7]), mx.array(y[:7]), arrays, w, c


@native
def test_fresh_output_ownership_survives_reexecution_and_runtime_destruction(native_problem):
    mx, bound, x, y, params, w, c = native_problem
    bound.bind(x, y, params)
    first = bound.run()
    expected = [host(value).copy() for value in [first[0], first[1], *first[2].values()]]
    first_pointers = {name: value.data_ptr() for name, value in bound.output_values.items()}
    bound.bind(x + mx.ones_like(x), y, params)
    bound.run()
    assert all(
        value.data_ptr() != first_pointers[name] for name, value in bound.output_values.items()
    )
    for value, saved in zip([first[0], first[1], *first[2].values()], expected, strict=True):
        np.testing.assert_array_equal(host(value), saved)
    # Use a separate adapter so no fixture retains its session or binding.
    graph, interface = gradient_model(w, c, batch=7, include_prediction=True)
    temporary = MlxDLPack(session(graph), interface)
    temporary.bind(x, y, params)
    prediction, loss, grads = temporary.run()
    del temporary
    gc.collect()
    oracle = reference(host(x), host(y), {n: host(v) for n, v in params.items()}, w, c)
    close(host(prediction), oracle[0])
    close(host(loss), oracle[1])
    for name in params:
        close(host(grads[name] + mx.zeros_like(grads[name])), oracle[2][name])


@native
def test_same_object_indexed_update_rebinds_replaced_storage(native_problem):
    mx, bound, x, y, params, w, c = native_problem
    bound.bind(x, y, params)
    bound.run()
    original = params["lora.A"]
    old_view = np.asarray(original, copy=False)
    old = old_view.copy()
    original[0, 0] = mx.array(2, dtype=mx.float32)
    assert params["lora.A"] is original
    bound.bind(x, y, params)
    assert (
        bound.input_values["lora.A"].data_ptr() == np.asarray(original, copy=False).ctypes.data
    )
    # The exported old storage remains independently owned after replacement.
    np.testing.assert_array_equal(old_view, old)
    actual = bound.run()
    expected = reference(host(x), host(y), {n: host(v) for n, v in params.items()}, w, c)
    close(host(actual[0]), expected[0])
    for name in params:
        close(host(actual[2][name]), expected[2][name])


@native
def test_invalid_feeds_disable_execution(native_problem):
    mx, bound, x, y, params, _, _ = native_problem
    with pytest.raises(RuntimeError, match="Bind"):
        bound.execute()
    with pytest.raises(RuntimeError, match="Execute"):
        bound.import_outputs()
    invalid = [
        (host(x), y, params),
        (x.astype(mx.float16), y, params),
        (x[:6], y, params),
        (x, y[:6], params),
        (x, y, {"lora.A": params["lora.A"]}),
    ]
    for a, b, p in invalid:
        bound.bind(x, y, params)
        with pytest.raises(ValueError):
            bound.bind(a, b, p)
        with pytest.raises(RuntimeError, match="Bind"):
            bound.execute()
    noncontiguous = mx.ones((7, 32), dtype=mx.float32)[:, ::2]
    with pytest.raises((ValueError, BufferError), match=r"contiguous|copy"):
        bound.bind(noncontiguous, y, params)
    with pytest.raises(RuntimeError, match="Bind"):
        bound.execute()


@native
def test_complete_native_gpu_training_checkpoint(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 200
    assert report["executed_providers"] == ["MLXExecutionProvider"]
    assert report["parameter_and_state_dlpack_device"] == [8, 0]
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5


def test_cpu_session_is_rejected_before_optional_mlx_import():
    from examples.lora.train import session as cpu_session

    graph, interface = gradient_model(*problem()[:2], batch=3)
    executor = cpu_session(ir.to_proto(graph).SerializeToString())
    with pytest.raises(ValueError, match="MLXExecutionProvider"):
        MlxDLPack(executor, interface)
