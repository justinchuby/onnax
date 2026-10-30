# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Finite-precision derivative invariants at saturated logits."""

import math

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session

import onnax


def model(op, shape=(3,), axis=-1):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=shape)
    tape = ir.tape.Tape()
    attrs = {"axis": axis} if op in ("Softmax", "LogSoftmax") else {}
    y = tape.op(op, [x], attrs)
    y.name, y.dtype, y.shape = "y", x.dtype, x.shape
    return ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 18}, name="precision"),
        ir_version=10,
    )


@pytest.mark.parametrize("executor", [evaluator, runtime_session])
@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("margin", [18.0, 40.0, 80.0])
def test_confident_class_retains_gradient(executor, op, margin):
    graph = model(op)
    x = np.array([0, -margin, -margin], np.float32)
    seed = np.array([1, 0, 0], np.float32)
    transformed = onnax.vjp(graph, inputs=["x"], outputs=["y"])
    gradient = executor(transformed).run(
        None, {"x": x, transformed.graph.inputs[-1].name: seed}
    )[-1]
    q = math.exp(-margin) / (1 + 2 * math.exp(-margin))
    expected = np.array([2 * q, -q, -q], np.float32)
    if op == "Softmax":
        expected *= np.float32(1 / (1 + 2 * math.exp(-margin)))
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=0)
    assert gradient[0] > 0
    assert abs(float(gradient.sum(dtype=np.float64))) <= float(abs(gradient[0])) * 1e-6
    tangent = onnax.jvp(graph, inputs=["x"], outputs=["y"])
    actual = executor(tangent).run(None, {"x": x, tangent.graph.inputs[-1].name: seed})[-1]
    expected_jvp = expected if op == "Softmax" else np.array([2 * q, -1, -1], np.float32)
    np.testing.assert_allclose(actual, expected_jvp, rtol=2e-5, atol=0)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_axis_ties_singleton_and_shift(op, axis):
    for shape in [(2, 3), (1, 1)]:
        graph = model(op, shape, axis)
        derivative = onnax.jvp(graph)
        x = np.zeros(shape, np.float32)
        value = runtime_session(derivative).run(
            None, {"x": x, derivative.graph.inputs[-1].name: np.ones(shape, np.float32)}
        )[-1]
        np.testing.assert_array_equal(value, np.zeros(shape, np.float32))


@pytest.mark.parametrize("executor", [evaluator, runtime_session])
def test_softplus_negative_tail_and_hvp(executor):
    graph = model("Softplus")
    x = np.array([-40, -18, 0], np.float32)
    vjp = onnax.vjp(graph)
    feeds = {"x": x, vjp.graph.inputs[-1].name: np.ones(3, np.float32)}
    gradient = executor(vjp).run(None, feeds)[-1]
    expected = (np.exp(x.astype(np.float64)) / (1 + np.exp(x.astype(np.float64)))).astype(
        np.float32
    )
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=0)
    hvp = onnax.jvp(vjp, inputs=["x"], outputs=[vjp.graph.outputs[-1].name])
    feeds[hvp.graph.inputs[-1].name] = np.ones(3, np.float32)
    curvature = executor(hvp).run(None, feeds)[-1]
    np.testing.assert_allclose(curvature, expected * (1 - expected), rtol=2e-5, atol=0)


def test_onnxscript_preserves_tail_gradient():
    from onnxscript import optimizer

    graph = onnax.vjp(model("LogSoftmax"))
    graph = optimizer.optimize(graph)
    values = runtime_session(graph).run(
        None,
        {
            "x": np.array([0, -40, -40], np.float32),
            graph.graph.inputs[-1].name: np.array([-1, 0, 0], np.float32),
        },
    )[-1]
    tail = math.exp(-40)
    np.testing.assert_allclose(values, [-2 * tail, tail, tail], rtol=2e-5, atol=0)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
def test_batched_derivative_composition(op):
    derivative = onnax.vjp(model(op))
    batched = onnax.vmap(derivative, in_axes=(0, 0))
    x = np.array([[0, -18, -40], [0, -40, -80]], np.float32)
    seed = np.array([[1, 0, 0], [1, 0, 0]], np.float32)
    actual = runtime_session(batched).run(
        None,
        {
            batched.graph.inputs[0].name: x,
            batched.graph.inputs[1].name: seed,
        },
    )[-1]
    executor = runtime_session(derivative)
    expected = np.stack(
        [
            executor.run(None, {"x": a, derivative.graph.inputs[-1].name: b})[-1]
            for a, b in zip(x, seed, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0)
