# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Activation derivatives across attributes, extreme inputs, and composition."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import grad, jvp, vjp, vmap


@pytest.mark.parametrize(
    "op,attributes",
    [
        ("Elu", {}),
        ("Elu", {"alpha": 0.4}),
        ("Elu", {"alpha": 0.0}),
        ("Selu", {}),
        ("Selu", {"alpha": 0.8, "gamma": 1.2}),
        ("Selu", {"gamma": 0.0}),
    ],
)
@pytest.mark.parametrize("curvature", [False, True])
def test_activation_products_and_curvature(op, attributes, curvature):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x], attributes)
    values = np.array([[-1.2, -0.4, 0.2], [0.7, -0.8, 1.1]])
    if curvature:
        loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
        _hvp_check(_model([x], tape, [loss]), {"x": values})
    else:
        _products(_model([x], tape, [y]), {"x": values})


@pytest.mark.parametrize("op", ["Elu", "Selu"])
def test_activation_extreme_inputs_and_batched_gradients(op):
    x = _value("x", [5], dtype=ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    model = _model([x], tape, [loss])
    values = np.array([-40.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32)
    alpha = np.float32(1.0 if op == "Elu" else 1.6732631921768188)
    gamma = np.float32(1.0 if op == "Elu" else 1.0507010221481323)
    negative_slope = alpha * gamma * np.exp(np.minimum(values, 0))
    expected = np.where(values > 0, gamma, negative_slope)

    forward = jvp(model)
    reverse = vjp(model)
    forward_values = runtime_session(forward).run(
        None, {"x": values, forward.graph.inputs[-1].name: np.ones_like(values)}
    )
    reverse_values = runtime_session(reverse).run(
        None, {"x": values, reverse.graph.inputs[-1].name: np.array(1, dtype=np.float32)}
    )
    np.testing.assert_allclose(forward_values[-1], expected.sum(), rtol=2e-6)
    np.testing.assert_allclose(reverse_values[-1], expected, rtol=2e-6, atol=0)

    gradient = grad(model)
    hvp = jvp(gradient)
    actual = runtime_session(hvp).run(
        None, {"x": values, hvp.graph.inputs[-1].name: np.ones_like(values)}
    )
    # At zero, both derivative orders follow the documented nonpositive branch.
    expected_curvature = np.where(values > 0, 0, negative_slope)
    np.testing.assert_allclose(actual[-1], expected_curvature, rtol=2e-6, atol=0)
    assert all(np.isfinite(output).all() for output in actual)

    batched = vmap(gradient, in_axes=(0,))
    batch = np.stack([values, values[::-1]])
    result = runtime_session(batched).run(None, {"x": batch})[-1]
    np.testing.assert_allclose(result, np.stack([expected, expected[::-1]]), rtol=2e-6, atol=0)


@pytest.mark.parametrize(
    "op,attributes",
    [
        ("HardSigmoid", {}),
        ("HardSigmoid", {"alpha": -0.4, "beta": 0.3}),
        ("HardSigmoid", {"alpha": 0.0, "beta": 0.6}),
        ("HardSwish", {}),
    ],
)
@pytest.mark.parametrize("curvature", [False, True])
def test_hard_activation_products_and_curvature(op, attributes, curvature):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x], attributes)
    values = np.array([[-7.0, -1.2, 0.2], [0.7, 1.8, 7.0]])
    if curvature:
        loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
        _hvp_check(_model([x], tape, [loss]), {"x": values})
    else:
        _products(_model([x], tape, [y]), {"x": values})


@pytest.mark.parametrize("op", ["HardSigmoid", "HardSwish"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 22, 28])
def test_hard_activation_runtime_and_batched_hvp(op, dtype, opset):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    values = np.array([-1000.0, -4.0, -1.0, 0.0, 1.0, 4.0, 1000.0], dtype=np_dtype)
    alpha = np_dtype(np.float32(0.2 if op == "HardSigmoid" else 1 / 6))
    affine = alpha * values + np_dtype(0.5)
    interior = (affine > 0) & (affine < 1)
    slope = (
        np.where(interior, alpha, 0)
        if op == "HardSigmoid"
        else np.where(interior, 2 * alpha * values + 0.5, np.where(affine >= 1, 1, 0))
    )
    curvature = (
        np.zeros_like(values) if op == "HardSigmoid" else np.where(interior, 2 * alpha, 0)
    )
    x = _value("x", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    model = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(model).SerializeToString()
    gradient = grad(model)
    product = jvp(gradient)
    executor = (
        runtime_session
        if opset == 18 or (opset == 22 and dtype == ir.DataType.FLOAT)
        else evaluator
    )
    actual = executor(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
    )
    np.testing.assert_allclose(actual[-2], slope, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual[-1], curvature, rtol=2e-6, atol=1e-7)
    mapped = vmap(product, in_axes=(0, 0))
    batch = np.stack([values, values[::-1]])
    actual = executor(mapped).run(
        None, {"x": batch, mapped.graph.inputs[-1].name: np.ones_like(batch)}
    )
    np.testing.assert_allclose(
        actual[-1], np.stack([curvature, curvature[::-1]]), rtol=2e-6, atol=1e-7
    )
    assert ir.to_proto(model).SerializeToString() == before


def test_hard_sigmoid_exact_clipping_boundaries():
    # Power-of-two alpha makes both breakpoints exactly representable.
    x = _value("x", [5])
    tape = ir.tape.Tape()
    y = tape.op("HardSigmoid", [x], {"alpha": 0.25, "beta": 0.5})
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    values = np.array([-3.0, -2.0, 0.0, 2.0, 3.0])
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones(5)}
    )
    np.testing.assert_array_equal(actual[-2], [0, 0, 0.25, 0, 0])
    np.testing.assert_array_equal(actual[-1], np.zeros(5))


def test_hard_swish_float_clipping_boundaries():
    x = _value("x", [5], dtype=ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op("HardSwish", [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    values = np.array([-4.0, -3.0, 0.0, 3.0, 4.0], dtype=np.float32)
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
    )
    np.testing.assert_allclose(actual[-2], [0, 0, 0.5, 1, 1])
    np.testing.assert_allclose(actual[-1], [0, 0, np.float32(1 / 3), 0, 0])
