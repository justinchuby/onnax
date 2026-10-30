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


@pytest.mark.parametrize("op", ["Sigmoid", "Tanh"])
def test_saturated_float_gradients_and_hvp_retain_small_tails(op):
    values = np.array([-40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0], dtype=np.float32)
    x = _value("x", [7], dtype=ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
    )
    scale = 2 if op == "Tanh" else 1
    tail = np.exp(-scale * np.abs(values.astype(np.float64)))
    slope = (4 if op == "Tanh" else 1) * tail / (1 + tail) ** 2
    curvature = -scale * np.sign(values) * slope * (1 - tail) / (1 + tail)
    np.testing.assert_allclose(actual[-2], slope, rtol=3e-6, atol=0)
    np.testing.assert_allclose(actual[-1], curvature, rtol=3e-6, atol=0)
    assert np.all(actual[-2] > 0)


@pytest.mark.parametrize("op", ["Asin", "Acos", "Atan", "Asinh"])
def test_inverse_trigonometric_products_and_hvp(op):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    values = np.array([[-0.8, -0.3, 0.0], [0.2, 0.5, 0.9]])
    _products(_model([x], tape, [y]).clone(), {"x": values})
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    _hvp_check(_model([x], tape, [loss]), {"x": values})


@pytest.mark.parametrize("op", ["Asin", "Acos"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 28])
def test_inverse_trigonometric_batched_curvature(op, dtype, opset):
    x = _value("x", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    source = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    product = jvp(grad(source))
    mapped = vmap(product, in_axes=(0, 0))
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    values = np.array([[-0.99, -0.5, 0.0], [0.1, 0.6, 0.99]], dtype=np_dtype)
    executor = runtime_session if opset == 18 and dtype == ir.DataType.FLOAT else evaluator
    actual = executor(mapped).run(
        None, {"x": values, mapped.graph.inputs[-1].name: np.ones_like(values)}
    )
    sign = -1 if op == "Acos" else 1
    reference = values.astype(np.float64)
    np.testing.assert_allclose(actual[-2], sign / np.sqrt(1 - reference**2), rtol=3e-6)
    np.testing.assert_allclose(
        actual[-1], sign * reference / (1 - reference**2) ** 1.5, rtol=4e-6, atol=1e-7
    )
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
def test_atan_large_inputs_and_branch_boundary_hvp(dtype):
    x = _value("x", [7], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op("Atan", [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    large = 1e10 if dtype == ir.DataType.FLOAT else 1e100
    values = np.array([-large, -2.0, -1.0, 0.0, 1.0, 2.0, large], dtype=np_dtype)
    executor = runtime_session if dtype == ir.DataType.FLOAT else evaluator
    mapped = vmap(product, in_axes=(0, 0))
    batch = np.stack([values, -values])
    actual = executor(mapped).run(
        None, {"x": batch, mapped.graph.inputs[-1].name: np.ones_like(batch)}
    )
    reference = batch.astype(np.float64)
    slope = 1 / (1 + reference**2)
    curvature = -2 * (reference / (1 + reference**2)) * slope
    np.testing.assert_allclose(actual[-2], slope.astype(np_dtype), rtol=3e-6, atol=0)
    np.testing.assert_allclose(actual[-1], curvature.astype(np_dtype), rtol=3e-6, atol=0)
    assert all(np.isfinite(value).all() for value in actual)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
def test_softsign_large_input_gradient_and_curvature(dtype):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    magnitude = 1e10 if dtype == ir.DataType.FLOAT else 1e100
    values = np.array([-magnitude, -1.0, 0.0, 1.0, magnitude], dtype=np_dtype)
    x = _value("x", [5], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op("Softsign", [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    executor = runtime_session if dtype == ir.DataType.FLOAT else evaluator
    actual = executor(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
    )
    inverse = 1 / (1 + np.abs(values.astype(np.float64)))
    np.testing.assert_allclose(
        actual[-2], (inverse * inverse).astype(np_dtype), rtol=3e-6, atol=0
    )
    curvature = -2 * np.sign(values) * inverse * inverse * inverse
    np.testing.assert_allclose(actual[-1], curvature.astype(np_dtype), rtol=3e-6, atol=0)
    assert np.all(actual[-2] > 0)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
def test_erf_saturated_finite_inputs_do_not_produce_nan_hvp(dtype):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    limit = np.finfo(np_dtype).max
    values = np.array([-limit, -1e30, 0.0, 1e30, limit], dtype=np_dtype)
    x = _value("x", [5], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op("Erf", [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    product = jvp(grad(_model([x], tape, [loss])))
    executor = runtime_session if dtype == ir.DataType.FLOAT else evaluator
    with np.errstate(over="ignore", invalid="ignore"):
        actual = executor(product).run(
            None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
        )
    np.testing.assert_allclose(actual[-2], [0, 0, 2 / np.sqrt(np.pi), 0, 0], rtol=1e-6)
    np.testing.assert_array_equal(actual[-1], np.zeros_like(values))
    assert all(np.isfinite(value).all() for value in actual)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 28])
def test_asinh_extremes_serialized_batched_hvp(dtype, opset, tmp_path):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    large = 1e30 if dtype == ir.DataType.FLOAT else 1e200
    curved = 1e18 if dtype == ir.DataType.FLOAT else 1e150
    tiny = 1e-18 if dtype == ir.DataType.FLOAT else 1e-150
    values = np.array(
        [-large, -curved, -2, -1, -tiny, 0, tiny, 1, 2, curved, large], dtype=np_dtype
    )
    x = _value("x", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op("Asinh", [x])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    source = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    ir.save(grad(source), tmp_path / "gradient.onnx")
    product = jvp(ir.load(tmp_path / "gradient.onnx"))
    mapped = vmap(product, in_axes=(0, 0))
    batch = np.stack([values, -values])
    executor = runtime_session if opset == 18 and dtype == ir.DataType.FLOAT else evaluator
    actual = executor(mapped).run(
        None, {"x": batch, mapped.graph.inputs[-1].name: np.ones_like(batch)}
    )
    reference = batch.astype(np.float64)
    norm = np.hypot(1, reference)
    slope = 1 / norm
    curvature = -(reference / norm) / norm / norm
    np.testing.assert_allclose(actual[-2], slope.astype(np_dtype), rtol=3e-6, atol=0)
    np.testing.assert_allclose(actual[-1], curvature.astype(np_dtype), rtol=3e-6, atol=0)
    assert all(np.isfinite(value).all() for value in actual)
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("op", ["Acosh", "Atanh"])
def test_inverse_hyperbolic_products_and_nonlinear_hvp(op):
    values = (
        np.array([[1.1, 1.4, 2.0], [2.5, 3.0, 5.0]])
        if op == "Acosh"
        else np.array([[-0.8, -0.3, 0.0], [0.2, 0.5, 0.9]])
    )
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    source = _model([x], tape, [y])
    _products(source, {"x": values})
    loss_tape = ir.tape.Tape(source.graph)
    loss = loss_tape.op("ReduceSum", [loss_tape.op("Mul", [y, y])], {"keepdims": 0})
    source.graph.outputs.clear()
    source.graph.outputs.append(loss)
    loss.name = "loss"
    _hvp_check(source, {"x": values})


@pytest.mark.parametrize("op", ["Acosh", "Atanh"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 28])
def test_inverse_hyperbolic_boundaries_serialized_batched_hvp(op, dtype, opset, tmp_path):
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    if op == "Acosh":
        boundary = np.nextafter(np_dtype(1), np_dtype(2))
        curved = 1e18 if dtype == ir.DataType.FLOAT else 1e150
        large = 1e30 if dtype == ir.DataType.FLOAT else 1e200
        values = np.array([boundary, 1.01, 2, curved, large], dtype=np_dtype)
    else:
        boundary = np.nextafter(np_dtype(1), np_dtype(0))
        tiny = 1e-18 if dtype == ir.DataType.FLOAT else 1e-150
        values = np.array([-boundary, -0.5, -tiny, 0, tiny, 0.5, boundary], dtype=np_dtype)
    x = _value("x", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    loss = tape.op("ReduceSum", [tape.op(op, [x])], {"keepdims": 0})
    source = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    ir.save(grad(source), tmp_path / "gradient.onnx")
    product = jvp(ir.load(tmp_path / "gradient.onnx"))
    mapped = vmap(product, in_axes=(0, 0))
    batch = np.stack([values, values])
    directions = np.stack([np.ones_like(values), -np.ones_like(values)])
    executor = runtime_session if opset == 18 and dtype == ir.DataType.FLOAT else evaluator
    actual = executor(mapped).run(None, {"x": batch, mapped.graph.inputs[-1].name: directions})
    reference = batch.astype(np.float64)
    if op == "Acosh":
        slope = (1 / np.sqrt(reference - 1)) / np.sqrt(reference + 1)
        curvature = -0.5 * slope * (1 / (reference - 1) + 1 / (reference + 1))
    else:
        slope = 1 / ((1 - reference) * (1 + reference))
        curvature = 2 * reference * slope * slope
    np.testing.assert_allclose(actual[-2], slope.astype(np_dtype), rtol=3e-6, atol=0)
    np.testing.assert_allclose(
        actual[-1], (curvature * directions).astype(np_dtype), rtol=3e-6, atol=0
    )
    assert all(np.isfinite(value).all() for value in actual)
    assert ir.to_proto(source).SerializeToString() == before
