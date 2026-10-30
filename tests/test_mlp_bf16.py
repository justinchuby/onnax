# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mixed-precision classifier exports, stable loss, analytic gradients and updates."""

import json
import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora.bf16 import round_bf16
from examples.lora.bf16_rules import lower_matmuls
from examples.lora.train import close
from examples.mlp import NAMES, SHAPES, model
from examples.mlp_bf16 import executor, graph, problem, reference, run


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_complete_reference_training_state_and_export(tmp_path, precision):
    report = run(tmp_path, precision=precision, measure=False)
    assert report["steps"] == 150
    assert report["parameter_bytes"] == 396 and report["state_bytes"] == 792
    assert report["validation_accuracy"] >= 0.95
    assert report["final_train_loss"] < report["initial_train_loss"] * 0.5
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
    assert report["checkpoint_resume_verified"]
    assert report["same_parameter_acceptance"]
    assert report["shared_gradient_update_acceptance"]
    if precision == "fp32":
        assert report["strict_numerical_acceptance"]
    manifest = json.loads((tmp_path / "interface.json").read_text())
    assert [p["input"] for p in manifest["parameters"]] == list(NAMES)
    assert [p["shape"] for p in manifest["parameters"]] == list(SHAPES)
    initial = problem()[-1]
    with np.load(tmp_path / "checkpoint.npz") as checkpoint:
        assert int(checkpoint["adam.step"]) == 150
        for name in NAMES:
            assert checkpoint[name].dtype == np.float32
            assert not np.array_equal(checkpoint[name], initial[name])
            for prefix in ("adam.first.", "adam.second."):
                assert checkpoint[prefix + name].dtype == np.float32
                assert np.isfinite(checkpoint[prefix + name]).all()
    for filename in (
        "gradient.raw.onnx",
        "gradient.onnx",
        "validation.onnx",
        "loss.onnx",
        "gradient.dynamic.onnx",
    ):
        loaded = ir.load(tmp_path / filename)
        onnx.checker.check_model(ir.to_proto(loaded), full_check=True)
        assert not loaded.functions and all(not n.domain for n in loaded.graph)
        assert set(loaded.graph.initializers).isdisjoint(NAMES)
        assert all(
            v.dtype == ir.DataType.FLOAT for v in (*loaded.graph.inputs, *loaded.graph.outputs)
        )
        products = [
            n
            for n in loaded.graph
            if n.op_type == "MatMul" and n.outputs[0].dtype == ir.DataType.BFLOAT16
        ]
        assert len(products) == (2 if precision == "bf16" else 0)


@pytest.mark.parametrize("batch", [1, 7, 192])
def test_raw_and_optimized_dynamic_exports_agree(batch):
    x, y, _, _, p = problem()
    feeds = {"features": x[:batch], "targets": y[:batch], **p}
    values = []
    for optimize in (False, True):
        g = graph(optimize=optimize)
        values.append(ReferenceEvaluator(ir.to_proto(g)).run(None, feeds))
    expected = reference(x[:batch], y[:batch], p)
    for result in values:
        for a, b in zip(result, [*expected[:2], *expected[2].values()], strict=True):
            close(a, b)
    for a, b in zip(*values, strict=True):
        close(a, b)


@pytest.mark.parametrize(
    "backend",
    [
        "reference",
        pytest.param(
            "mlx",
            marks=pytest.mark.skipif(
                os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in"
            ),
        ),
    ],
)
def test_fp32_loss_stays_stable_with_bf16_matrix_compute(backend):
    g = graph(batch=2)
    evaluator = executor(g, backend)
    params = {n: np.zeros(s, np.float32) for n, s in zip(NAMES, SHAPES, strict=True)}
    params[NAMES[-1]] = np.array([-1000, 0, 1000], np.float32)
    feeds = {
        "features": np.zeros((2, 2), np.float32),
        "targets": np.array([[1, 0, 0], [1, 0, 0]], np.float32),
    }
    first = evaluator.run(None, {**feeds, **params})
    params[NAMES[-1]] += np.float32(10000)
    second = evaluator.run(None, {**feeds, **params})
    for values in (first, second):
        assert all(np.isfinite(v).all() for v in values)
        close(values[1], np.array(2000, np.float32))
        close(values[-1], np.array([-1, 0, 1], np.float32))
    for a, b in zip(first[1:], second[1:], strict=True):
        close(a, b)


def test_bf16_analytic_gradients_match_constant_rounding_offset_surrogate():
    rng = np.random.default_rng(34)
    params = {
        n: rng.normal(0, 0.2, s).astype(np.float32) for n, s in zip(NAMES, SHAPES, strict=True)
    }
    x = rng.normal(size=(5, 2)).astype(np.float32)
    targets = np.eye(3, dtype=np.float32)[[0, 1, 2, 1, 0]]
    qx, qw1, qw2 = (
        round_bf16(v).astype(np.float64) for v in (x, params[NAMES[0]], params[NAMES[2]])
    )
    z0 = (
        round_bf16(qx.astype(np.float32) @ qw1.astype(np.float32)) + params[NAMES[1]]
    ).astype(np.float64)
    h0 = np.tanh(z0.astype(np.float32)).astype(np.float64)
    qh = round_bf16(h0.astype(np.float32)).astype(np.float64)
    logits0 = (
        round_bf16(qh.astype(np.float32) @ qw2.astype(np.float32)) + params[NAMES[3]]
    ).astype(np.float64)

    def smooth(candidate):
        w1 = qw1 + (candidate[NAMES[0]] - params[NAMES[0]].astype(np.float64))
        z = z0 + qx @ (w1 - qw1) + (candidate[NAMES[1]] - params[NAMES[1]].astype(np.float64))
        hidden = qh + np.tanh(z) - np.tanh(z0)
        w2 = qw2 + (candidate[NAMES[2]] - params[NAMES[2]].astype(np.float64))
        logits = (
            logits0
            + (hidden @ w2 - qh @ qw2)
            + (candidate[NAMES[3]] - params[NAMES[3]].astype(np.float64))
        )
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return -np.mean(np.sum(targets * logp, axis=-1))

    analytic = reference(x, targets, params)[2]
    for name in NAMES:
        for coordinate in ((0,) * params[name].ndim, tuple(d - 1 for d in params[name].shape)):
            plus = {n: p.astype(np.float64) for n, p in params.items()}
            minus = {n: p.astype(np.float64) for n, p in params.items()}
            plus[name][coordinate] += 1e-5
            minus[name][coordinate] -= 1e-5
            close((smooth(plus) - smooth(minus)) / 2e-5, analytic[name][coordinate])


def test_lowering_preserves_integer_loss_controls_and_rejects_integer_matrices():
    source = model()
    before = ir.to_proto(source).SerializeToString()
    mixed = lower_matmuls(source)
    assert mixed.graph.initializers["class_axis"].dtype == ir.DataType.INT64
    assert ir.to_proto(source).SerializeToString() == before
    source.graph.inputs[2].dtype = ir.DataType.INT64
    with pytest.raises(TypeError, match="FP32"):
        lower_matmuls(source)
    source = model()
    weight = source.graph.inputs[2]
    source.graph.inputs.remove(weight)
    weight.dtype = ir.DataType.INT64
    weight.const_value = ir.tensor(np.ones((2, 16), np.int64))
    source.graph.register_initializer(weight)
    with pytest.raises(TypeError, match="matrix operands"):
        lower_matmuls(source)


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1",
    reason="Set ONNAX_TEST_MLX_EP=1 on Apple Silicon",
)
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_complete_native_classification_and_checkpoint(tmp_path, precision):
    report = run(tmp_path, backend="mlx", precision=precision, measure=False)
    assert report["executed_providers"] == ["MLXExecutionProvider"]
    assert report["checkpoint_resume_verified"] and report["validation_accuracy"] >= 0.95
    assert report["strict_numerical_acceptance"] == (not report["errors"])
    g = ir.load(tmp_path / "gradient.onnx")
    assert not {n.op_type for n in g.graph} & {"ScatterElements", "Reshape"}
