# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Named shared bindings validate metadata, retain owners and invalidate failed feeds."""

import gc
import os
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import host
from examples.lora.train import close
from examples.mlp import NAMES
from examples.mlp_bf16 import graph, problem
from examples.mlx_binding import MlxBinding, static_shapes

native = pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in"
)


@pytest.mark.parametrize("shape", [None, [None, 3], ["batch", 3], [0, 3], [-1], [True]])
def test_binding_requires_static_positive_dimensions(shape):
    with pytest.raises(ValueError, match="static positive"):
        static_shapes([SimpleNamespace(name="value", shape=shape, type="tensor(float)")])


def test_binding_metadata_accepts_scalars_and_rejects_other_dtypes():
    assert static_shapes([SimpleNamespace(name="loss", shape=[], type="tensor(float)")]) == {
        "loss": ()
    }
    with pytest.raises(ValueError, match="FP32"):
        static_shapes([SimpleNamespace(name="labels", shape=[3], type="tensor(int64)")])


def test_cpu_session_is_rejected_before_mlx_import():
    g = graph(precision="fp32", batch=7, with_gradients=False)
    executor = ort.InferenceSession(
        ir.to_proto(g).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    with pytest.raises(ValueError, match="MLXExecutionProvider"):
        MlxBinding(executor)


@native
def test_arbitrary_parameter_names_selected_output_order_and_retained_ownership():
    import mlx.core as mx

    x, y, _, _, p = problem()
    executor = session(graph(precision="fp32", batch=7))
    names = [executor.get_outputs()[-1].name, "loss", "logits"]
    expected = executor.run(names, {"features": x[:7], "targets": y[:7], **p})
    with mx.stream(mx.gpu):
        feeds = {n: mx.array(v) for n, v in {"features": x[:7], "targets": y[:7], **p}.items()}
        binding = MlxBinding(executor, outputs=names)
        binding.bind(feeds)
        first = binding.run()
        assert list(first) == names
        for n, v in zip(names, expected, strict=True):
            close(host(first[n]), v)
        snapshots = {n: host(v).copy() for n, v in first.items()}
        pointers = {n: v.data_ptr() for n, v in binding.output_values.items()}
        feeds[NAMES[0]] = feeds[NAMES[0]] + 0.1
        binding.bind(feeds)
        second = binding.run()
        assert all(binding.output_values[n].data_ptr() != pointers[n] for n in names)
        assert not np.array_equal(host(first["logits"]), host(second["logits"]))
        del binding, executor, second, feeds
        gc.collect()
        for n, v in first.items():
            np.testing.assert_array_equal(host(v), snapshots[n])


@native
def test_failed_generic_feeds_disable_execution_and_output_import():
    import mlx.core as mx

    x, y, _, _, p = problem()
    executor = session(graph(precision="fp32", batch=7))
    with mx.stream(mx.gpu):
        feeds = {n: mx.array(v) for n, v in {"features": x[:7], "targets": y[:7], **p}.items()}
        binding = MlxBinding(executor)
        bad = [
            {n: v for n, v in feeds.items() if n != NAMES[0]},
            {**feeds, "targets": feeds["targets"][:6]},
            {**feeds, NAMES[0]: feeds[NAMES[0]].astype(mx.float16)},
            {**feeds, "features": x[:7]},
        ]
        for invalid in bad:
            binding.bind(feeds)
            with pytest.raises(ValueError):
                binding.bind(invalid)
            with pytest.raises(RuntimeError, match="Bind"):
                binding.execute()
            with pytest.raises(RuntimeError, match="Execute"):
                binding.import_outputs()
        for outputs in ([], ["missing"], ["loss", "loss"]):
            with pytest.raises(ValueError, match="output names"):
                MlxBinding(executor, outputs=outputs)


@native
def test_scalar_input_uses_shared_storage():
    import mlx.core as mx

    x = ir.val("position", dtype=ir.DataType.FLOAT, shape=[3])
    shift = ir.val("shift", dtype=ir.DataType.FLOAT, shape=[])
    tape = ir.tape.Tape()
    y = tape.op("Add", [x, shift])
    y.name, y.dtype, y.shape = "result", ir.DataType.FLOAT, ir.Shape([3])
    model = ir.Model(
        ir.Graph([x, shift], [y], nodes=tape.nodes, opset_imports={"": 18}), ir_version=10
    )
    with mx.stream(mx.gpu):
        binding = MlxBinding(session(model))
        binding.bind(
            {
                "position": mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                "shift": mx.array(0.25, dtype=mx.float32),
            }
        )
        close(host(binding.run()["result"]), np.array([1.25, 2.25, 3.25], np.float32))
