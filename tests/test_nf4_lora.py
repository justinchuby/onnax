# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Packed NF4 math, frozen-base derivatives and complete external LoRA training."""

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import optimizer

import onnax
from examples.lora.model import PARAMETERS, adapt, base_model, with_loss
from examples.lora.nf4 import CODEBOOK, dequantize, lower, quantize
from examples.lora.nf4_train import run
from examples.lora.train import problem, reference, session


@pytest.mark.parametrize("shape", [(3, 5), (16, 24), (1, 1)])
def test_packing_tail_and_error_bound(shape):
    rng = np.random.default_rng(123)
    weight = rng.normal(size=shape).astype(np.float32)
    packed, scales = quantize(weight)
    result = dequantize(packed, scales, shape)
    assert packed.dtype == np.uint8 and scales.dtype == np.float32
    assert packed.nbytes == (weight.size + 63) // 64 * 32
    flat = weight.reshape(-1)
    for i, value in enumerate(flat):
        scale = scales[i // 64]
        expected = CODEBOOK[np.argmin(np.abs(value / scale - CODEBOOK))] * scale
        assert result.flat[i] == expected
    zero = np.zeros(shape, np.float32)
    q, s = quantize(zero)
    assert np.all(q == 0x77) and not np.any(s)
    np.testing.assert_array_equal(dequantize(q, s, shape), zero)


@pytest.mark.parametrize("size", [0, 1, 3, -2, True, 64.0])
def test_invalid_block_size(size):
    with pytest.raises(ValueError, match="even block"):
        quantize(np.ones((2, 2), np.float32), block_size=size)
    with pytest.raises(ValueError, match="even block"):
        dequantize(np.zeros(32, np.uint8), np.ones(1, np.float32), (2, 2), block_size=size)


def test_invalid_weights_buffers_and_names():
    for weight in [
        np.zeros(2, np.float32),
        np.empty((0, 2), np.float32),
        np.ones((2, 2), np.float64),
        np.full((2, 2), np.nan, np.float32),
    ]:
        with pytest.raises(ValueError, match="FP32 weight"):
            quantize(weight)
    q, s = quantize(np.ones((2, 2), np.float32))
    for packed, scale in [
        (q[:-1], s),
        (q.astype(np.int64), s),
        (q, -s),
        (q, s.astype(np.float64)),
        (q, np.array([np.nan], np.float32)),
    ]:
        with pytest.raises(ValueError, match="NF4 buffers"):
            dequantize(packed, scale, (2, 2))
    w, c, *_ = problem()
    model = adapt(base_model(w, c))
    before = ir.to_proto(model).SerializeToString()
    for names in [[], ["base.W", "base.W"], "base.W", ["absent"]]:
        with pytest.raises(ValueError):
            lower(model, names=names)
    model.graph.inputs.append(ir.val("base.W.nf4", dtype=ir.DataType.FLOAT, shape=[1]))
    with pytest.raises(ValueError, match="collision"):
        lower(model)
    model.graph.inputs.pop()
    assert ir.to_proto(model).SerializeToString() == before


def test_frozen_head_input_gradient_and_source_preservation():
    w, c, x, y, _, _, params = problem()
    params[PARAMETERS[1]][:] = np.float32(0.05)
    source = with_loss(adapt(base_model(w, c)))
    before = ir.to_proto(source).SerializeToString()
    quantized, info = lower(source)
    gradient = onnax.grad(quantized, inputs=["features", *PARAMETERS], output="loss")
    optimized = optimizer.optimize(gradient.clone(), inline=False)
    assert ir.to_proto(source).SerializeToString() == before
    assert len(optimized.functions) >= 3
    for name in info:
        assert name not in optimized.graph.initializers
    weights = {
        n: dequantize(
            quantized.graph.initializers[v["packed"]].const_value.numpy(),
            quantized.graph.initializers[v["scales"]].const_value.numpy(),
            v["shape"],
        )
        for n, v in info.items()
    }
    feeds = {"features": x[:3], "target": y[:3], **params}
    result = session(ir.to_proto(optimized).SerializeToString()).run(None, feeds)
    a, b = (params[n] for n in PARAMETERS)
    hidden = np.tanh(x[:3] @ weights["base.W"] + np.float32(0.25) * (x[:3] @ a @ b))
    delta = (np.float32(2 / 24) * (result[0] - y[:3]) @ weights["base.C"].T) * (
        1 - hidden * hidden
    )
    dx = delta @ weights["base.W"].T + np.float32(0.25) * delta @ b.T @ a.T
    np.testing.assert_allclose(result[2], dx, rtol=1e-4, atol=1e-6)
    expected = reference(x[:3], y[:3], params, weights["base.W"], weights["base.C"])
    for actual, wanted in zip(result[3:], expected[2].values(), strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="frozen"):
        onnax.grad(quantized, inputs=[info["base.C"]["scales"]], output="loss")


def test_runtime_optimization_disabled_preserves_packed_storage(tmp_path):
    w, c, *_ = problem()
    quantized, _ = lower(with_loss(adapt(base_model(w, c))))
    gradient = optimizer.optimize(
        onnax.grad(quantized, inputs=list(PARAMETERS), output="loss"), inline=False
    )
    path = tmp_path / "resolved.onnx"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.optimized_model_filepath = str(path)
    options.intra_op_num_threads = 1
    ort.InferenceSession(
        ir.to_proto(gradient).SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    resolved = onnx.load(path)
    assert any(v.name == "base.W.nf4" for v in resolved.graph.initializer)
    assert not any(
        v.data_type == onnx.TensorProto.FLOAT
        and list(v.dims) in ([16, 24], [24, 8], [8, 24], [24, 16])
        for v in resolved.graph.initializer
    )


def test_complete_nf4_lora_training(tmp_path):
    report = run(tmp_path, measure=False)
    assert report["steps"] == 200
    assert report["train_loss"]["final"] < report["train_loss"]["initial"] * 0.5
    assert report["validation_loss"]["final"] < report["validation_loss"]["initial"] * 0.5
    assert report["frozen_weights_unchanged"] and report["adapter_checkpoint_replay"]
    assert report["base_packed_bytes"] == 324 and report["base_original_bytes"] == 2304


def test_nf4_hvp_matches_independent_gradient_directional_difference():
    w, c, x, y, _, _, params = problem()
    params[PARAMETERS[1]][:] = np.float32(0.03)
    model, info = lower(with_loss(adapt(base_model(w, c))))
    gradient = onnax.grad(model, inputs=list(PARAMETERS), output="loss")
    hvp = onnax.jvp(gradient, inputs=list(PARAMETERS))
    direction = {n: np.full_like(p, 0.02) for n, p in params.items()}
    feeds = {
        "features": x[:3],
        "target": y[:3],
        **params,
        **{n + "_tangent": d for n, d in direction.items()},
    }
    values = session(ir.to_proto(hvp).SerializeToString()).run(None, feeds)
    weights = {
        n: dequantize(
            model.graph.initializers[v["packed"]].const_value.numpy(),
            model.graph.initializers[v["scales"]].const_value.numpy(),
            v["shape"],
        ).astype(np.float64)
        for n, v in info.items()
    }
    eps = 1e-4
    plus = {
        n: p.astype(np.float64) + eps * direction[n].astype(np.float64)
        for n, p in params.items()
    }
    minus = {
        n: p.astype(np.float64) - eps * direction[n].astype(np.float64)
        for n, p in params.items()
    }
    a = reference(
        x[:3].astype(np.float64),
        y[:3].astype(np.float64),
        plus,
        weights["base.W"],
        weights["base.C"],
    )[2]
    b = reference(
        x[:3].astype(np.float64),
        y[:3].astype(np.float64),
        minus,
        weights["base.W"],
        weights["base.C"],
    )[2]
    for actual, n in zip(values[-2:], PARAMETERS, strict=True):
        np.testing.assert_allclose(actual, (a[n] - b[n]) / (2 * eps), rtol=1e-4, atol=1e-6)


def test_loaded_nf4_forward_can_generate_gradients(tmp_path):
    from examples.lora.nf4 import register_rules

    w, c, *_ = problem()
    model, _ = lower(adapt(base_model(w, c)))
    path = tmp_path / "nf4.onnx"
    ir.save(model, path)
    loaded = ir.load(path)
    before = ir.to_proto(loaded).SerializeToString()
    register_rules(loaded)
    gradient = onnax.grad(with_loss(loaded), inputs=list(PARAMETERS), output="loss")
    onnx.checker.check_model(ir.to_proto(gradient), full_check=True)
    assert ir.to_proto(loaded).SerializeToString() == before


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 4, 3)])
def test_nf4_linear_supports_vector_and_token_batches(shape):
    rng = np.random.default_rng(15)
    weights = rng.normal(size=(3, 5)).astype(np.float32)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=shape)
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(weights, name="weight"))
    y = tape.op("MatMul", [x, w])
    loss = tape.op("ReduceMean", [tape.op("Mul", [y, y])], {"keepdims": 0})
    loss.name = "loss"
    source = ir.Model(
        ir.Graph(
            [x],
            [loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="token_linear",
        ),
        ir_version=10,
    )
    lowered, _ = lower(source, names=("weight",))
    gradient = onnax.grad(lowered, inputs=["x"], output="loss")
    values = rng.normal(size=shape).astype(np.float32)
    actual = session(ir.to_proto(gradient).SerializeToString()).run(None, {"x": values})
    packed, scales = quantize(weights)
    decoded = dequantize(packed, scales, weights.shape)
    predicted = values @ decoded
    expected = (np.float32(2 / predicted.size) * predicted) @ decoded.T
    np.testing.assert_allclose(actual[0], np.mean(predicted * predicted), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected, rtol=1e-4, atol=1e-6)


def test_nf4_codebook_layout_is_high_nibble_first():
    packed, scales = quantize(CODEBOOK.reshape(4, 4))
    np.testing.assert_array_equal(
        packed[:8], np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF], np.uint8)
    )
    np.testing.assert_array_equal(scales, np.ones(1, np.float32))
    np.testing.assert_array_equal(dequantize(packed, scales, (4, 4)), CODEBOOK.reshape(4, 4))
