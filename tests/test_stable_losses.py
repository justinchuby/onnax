# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""FP32 loss primitives and their higher derivatives."""

import math
import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import ShapeInferencePass

import onnax
from examples.huggingface import stable_losses as losses


def loss_model(kind, *, vocab=3, opset=18):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, vocab])
    other = ir.val(
        "other",
        dtype=ir.DataType.INT64 if kind == "hard" else ir.DataType.FLOAT,
        shape=[2] if kind == "hard" else [2, vocab],
    )
    graph = ir.Graph([x, other], [], nodes=[], opset_imports={"": opset}, name="stable_loss")
    ctx = onnax.AutodiffContext(graph)
    rows = (losses.hard_cross_entropy if kind == "hard" else losses.distillation)(
        ctx, x, other
    )
    y = ctx.op("ReduceSum", rows, keepdims=0)
    y.name, y.dtype, y.shape = "loss", x.dtype, ir.Shape([])
    graph.outputs.append(y)
    model = ir.Model(graph, ir_version=10)
    losses.install(model)
    ShapeInferencePass()(model)
    return model


@pytest.mark.parametrize("margin", [18, 40, 80])
def test_hard_loss_preserves_fp32_tail(margin):
    model = onnax.grad(loss_model("hard"), inputs=["x"], output="loss")
    x = np.array([[0, -margin, -margin], [1, 2, 3]], np.float32)
    loss, gradient = runtime_session(model).run(
        None, {"x": x, "other": np.array([0, -100], np.int64)}
    )
    tail = math.exp(-margin)
    np.testing.assert_allclose(loss, math.log1p(2 * tail), rtol=2e-5, atol=0)
    np.testing.assert_allclose(gradient[0], [-2 * tail, tail, tail], rtol=2e-5, atol=0)
    np.testing.assert_array_equal(gradient[1], np.zeros(3, np.float32))


def test_distillation_identical_logits_and_curvature():
    model = onnax.grad(loss_model("soft"), inputs=["x"], output="loss")
    x = np.array([[0, -18, -40], [1, 2, 3]], np.float32)
    feeds = {"x": x, "other": x.copy()}
    np.testing.assert_array_equal(
        runtime_session(model).run(None, feeds)[-1], np.zeros_like(x)
    )
    product = onnax.jvp(model, inputs=["x"], outputs=[model.graph.outputs[-1].name])
    direction = np.array([[1, 0, 0], [1, -2, 3]], np.float32)
    feeds[product.graph.inputs[-1].name] = direction
    actual = runtime_session(product).run(None, feeds)[-1]
    p = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    expected = p * (direction - (p * direction).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-4, 0.2, 20.0])
def test_distillation_nearby_and_separated_logits(scale):
    model = onnax.grad(loss_model("soft"), inputs=["x"], output="loss")
    teacher = np.array([[0, -18, -40], [1, 2, 3]], np.float32)
    student = teacher + np.array([[1, -2, 3], [-1, 2, 0]], np.float32) * np.float32(scale)
    gradient = runtime_session(model).run(None, {"x": student, "other": teacher})[-1]

    # Independent host oracle; arrays supplied to the graph and its loss remain FP32.
    def probabilities(x):
        x = x.astype(np.float64)
        p = np.exp(x - x.max(axis=-1, keepdims=True))
        return p / p.sum(axis=-1, keepdims=True)

    expected = probabilities(student) - probabilities(teacher)
    # Recover the pivot from the small entries, including below double precision near one.
    for row in range(2):
        pivot = student[row].argmax()
        expected[row, pivot] = -np.delete(expected[row], pivot).sum()
    np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=0)


@pytest.mark.parametrize("compute_dtype", ["float16", "bfloat16"])
def test_rounded_compute_logits_with_fp32_loss(compute_dtype):
    dtype = (
        np.float16 if compute_dtype == "float16" else pytest.importorskip("ml_dtypes").bfloat16
    )
    logits = (
        np.array([[0, -18.12, -40.12], [4.03, 2.01, 1.01]], np.float32)
        .astype(dtype)
        .astype(np.float32)
    )
    labels = np.array([0, 2], np.int64)
    model = onnax.grad(loss_model("hard"), inputs=["x"], output="loss")
    value, gradient = runtime_session(model).run(None, {"x": logits, "other": labels})
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    expected = p.copy()
    expected[np.arange(2), labels] = 0
    expected[np.arange(2), labels] = -expected.sum(axis=-1)
    assert value.dtype == gradient.dtype == np.float32
    assert gradient[0, 0] != 0
    np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=0)


def test_distillation_batching_composes_with_gradient():
    model = onnax.grad(loss_model("soft"), inputs=["x"], output="loss")
    batched = onnax.vmap(model, in_axes=(0, 0))
    x = np.array([[[0, -18, -40], [1, 2, 3]], [[1, 1, 1], [0, -40, -80]]], np.float32)
    feeds = {"x": x, "other": x.copy()}
    np.testing.assert_array_equal(
        runtime_session(batched).run(None, feeds)[-1], np.zeros_like(x)
    )


def test_hard_loss_long_vocabulary_other_pivot_gradient():
    vocab = 151936
    model = onnax.grad(loss_model("hard", vocab=vocab), inputs=["x"], output="loss")
    x = np.full((2, vocab), -4, np.float32)
    x[0, 0] = 10
    value, gradient = runtime_session(model).run(
        None, {"x": x, "other": np.array([0, -100], np.int64)}
    )
    mass = (vocab - 1) * math.exp(-14)
    np.testing.assert_allclose(value, math.log1p(mass), rtol=2e-5, atol=0)
    np.testing.assert_allclose(gradient[0, 0], -mass / (1 + mass), rtol=2e-5, atol=0)
    np.testing.assert_allclose(gradient[0, 1:], math.exp(-14) / (1 + mass), rtol=2e-5, atol=0)


def test_hard_loss_ignored_nan_rows_have_zero_derivatives():
    model = onnax.grad(loss_model("hard"), inputs=["x"], output="loss")
    x = np.full((2, 3), np.nan, np.float32)
    value, gradient = runtime_session(model).run(
        None, {"x": x, "other": np.full(2, -100, np.int64)}
    )
    np.testing.assert_array_equal(value, np.float32(0))
    np.testing.assert_array_equal(gradient, np.zeros_like(x))


def test_hard_loss_jvp_hvp_and_batching():
    primal = loss_model("hard")
    gradient = onnax.grad(primal, inputs=["x"], output="loss")
    tangent = onnax.jvp(primal, inputs=["x"], outputs=["loss"])
    hessian = onnax.jvp(gradient, inputs=["x"], outputs=[gradient.graph.outputs[-1].name])
    x = np.array([[0.2, -0.1, 0.4], [1, 2, 3]], np.float32)
    labels = np.array([1, -100], np.int64)
    direction = np.array([[1, -2, 0.5], [0.5, 1, 2]], np.float32)
    probability = np.exp(x.astype(np.float64) - x.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    expected_gradient = probability.copy()
    expected_gradient[0, 1] -= 1
    expected_gradient[1] = 0
    expected_hessian = probability * (
        direction - (probability * direction).sum(-1, keepdims=True)
    )
    expected_hessian[1] = 0
    feeds = {"x": x, "other": labels}
    jv = runtime_session(tangent).run(
        None, {**feeds, tangent.graph.inputs[-1].name: direction}
    )[-1]
    hv = runtime_session(hessian).run(
        None, {**feeds, hessian.graph.inputs[-1].name: direction}
    )[-1]
    np.testing.assert_allclose(jv, (expected_gradient * direction).sum(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(hv, expected_hessian, rtol=1e-5, atol=1e-7)
    batched = onnax.vmap(gradient, in_axes=(0, 0))
    values = runtime_session(batched).run(
        None, {"x": np.stack([x, x]), "other": np.stack([labels, labels])}
    )[-1]
    np.testing.assert_allclose(
        values, np.stack([expected_gradient, expected_gradient]), rtol=1e-5, atol=1e-7
    )


def test_hard_loss_single_class():
    model = onnax.grad(loss_model("hard", vocab=1), inputs=["x"], output="loss")
    values = runtime_session(model).run(
        None, {"x": np.array([[7], [-3]], np.float32), "other": np.array([0, -100], np.int64)}
    )
    np.testing.assert_array_equal(values[0], np.float32(0))
    np.testing.assert_array_equal(values[1], np.zeros((2, 1), np.float32))


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
def test_native_hard_loss_long_vocabulary_and_ignored_nan():
    from examples.lora.mlx_runtime import register_library, session

    library = os.environ.get("ONNAX_MLX_EP_LIBRARY")
    if library:
        register_library(library)
    vocab = 151936
    model = onnax.grad(loss_model("hard", vocab=vocab), inputs=["x"], output="loss")
    x = np.full((2, vocab), -4, np.float32)
    x[0, 0] = 10
    x[1] = np.nan
    value, gradient = session(model).run(
        None, {"x": x, "other": np.array([0, -100], np.int64)}
    )
    mass = (vocab - 1) * math.exp(-14)
    np.testing.assert_allclose(value, math.log1p(mass), rtol=2e-5, atol=0)
    np.testing.assert_allclose(gradient[0, 0], -mass / (1 + mass), rtol=2e-5, atol=0)
    np.testing.assert_allclose(gradient[0, 1:], math.exp(-14) / (1 + mass), rtol=2e-5, atol=0)
    np.testing.assert_array_equal(gradient[1], np.zeros(vocab, np.float32))


def test_hard_loss_curvature_at_equal_binary_logits():
    gradient = onnax.grad(loss_model("hard", vocab=2), inputs=["x"], output="loss")
    hessian = onnax.jvp(gradient, inputs=["x"], outputs=[gradient.graph.outputs[-1].name])
    direction = np.array([[1, -1], [0, 0]], np.float32)
    result = runtime_session(hessian).run(
        None,
        {
            "x": np.zeros((2, 2), np.float32),
            "other": np.array([0, -100], np.int64),
            hessian.graph.inputs[-1].name: direction,
        },
    )[-1]
    np.testing.assert_allclose(result, [[0.5, -0.5], [0, 0]], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("in_axes", [(0, None), (None, 0)])
def test_hard_loss_batching_with_shared_operand(in_axes):
    gradient = onnax.grad(loss_model("hard"), inputs=["x"], output="loss")
    batched = onnax.vmap(gradient, in_axes=in_axes)
    x = np.array([[0.2, -0.1, 0.4], [1, 2, 3]], np.float32)
    labels = np.array([1, -100], np.int64)
    xs = np.stack([x, x + np.array([[-1, 0, 1], [0, 0, 0]], np.float32)])
    ys = np.stack([labels, np.array([0, 2], np.int64)])
    inputs = {"x": xs if in_axes[0] == 0 else x, "other": ys if in_axes[1] == 0 else labels}
    executor = runtime_session(gradient)
    expected = np.stack(
        [
            executor.run(
                None,
                {
                    "x": xs[i] if in_axes[0] == 0 else x,
                    "other": ys[i] if in_axes[1] == 0 else labels,
                },
            )[-1]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(
        runtime_session(batched).run(None, inputs)[-1], expected, rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("kind", ["hard", "soft"])
def test_loss_functions_follow_model_opset(kind):
    model = onnax.grad(loss_model(kind, opset=24), inputs=["x"], output="loss")
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    assert all(function.opset_imports[""] == 24 for function in model.functions.values())
    x = np.array([[0, -18, -40], [1, 2, 3]], np.float32)
    other = np.array([0, -100], np.int64) if kind == "hard" else x.copy()
    values = runtime_session(model).run(None, {"x": x, "other": other})
    assert all(np.isfinite(value).all() for value in values)


def test_stable_loss_rejects_older_model_opset():
    with pytest.raises(ValueError, match="opset 18 or newer"):
        loss_model("hard", opset=17)
