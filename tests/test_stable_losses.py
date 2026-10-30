# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""FP32 loss primitives and their higher derivatives."""

import math

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import ShapeInferencePass

import onnax
from examples.huggingface import stable_losses as losses


def loss_model(kind):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    other = ir.val(
        "other",
        dtype=ir.DataType.INT64 if kind == "hard" else ir.DataType.FLOAT,
        shape=[2] if kind == "hard" else [2, 3],
    )
    graph = ir.Graph([x, other], [], nodes=[], opset_imports={"": 18}, name="stable_loss")
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
