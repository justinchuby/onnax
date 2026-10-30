# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Composite reductions retain axes semantics under differentiation and batching."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import grad, jvp, vjp, vmap


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceSumSquare", "ReduceLogSum"])
@pytest.mark.parametrize("axes", [None, [], [-1], [0, 1]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
def test_products_axes_and_noop_semantics(op, axes, keepdims, noop):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    inputs = [x]
    if axes is not None:
        inputs.append(
            tape.op("Constant", [], {"value": ir.tensor(axes, dtype=ir.DataType.INT64)})
        )
    y = tape.op(op, inputs, {"keepdims": keepdims, "noop_with_empty_axes": noop})
    values = np.array([[-1.2, 0.4, -0.7], [0.3, -0.9, 1.1]])
    if op == "ReduceLogSum":
        values = np.abs(values)
    _products(_model([x], tape, [y]), {"x": values})


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceSumSquare", "ReduceLogSum"])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_runtime_axes_and_nonlinear_outer_hvp(op, keepdims):
    x, axes = _value("x", ["n", 3]), _value("axes", [1], dtype=ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op(op, [x, axes], {"keepdims": keepdims})
    y.shape = ir.Shape(["n", 1] if keepdims else ["n"])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    feeds = {
        "x": np.array([[-1.2, 0.4, -0.7], [0.3, -0.9, 1.1]]),
        "axes": np.array([-1], dtype=np.int64),
    }
    if op == "ReduceLogSum":
        feeds["x"] = np.abs(feeds["x"])
    product = _hvp_check(_model([x, axes], tape, [loss]), feeds, inputs=["x"])
    feeds[product.graph.inputs[-1].name] = np.ones_like(feeds["x"])
    expected = evaluator(product).run(None, feeds)
    actual = runtime_session(product).run(None, feeds)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceSumSquare"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("shape", [[], [0, 3], [2, 3]])
@pytest.mark.parametrize("opset", [18, 28])
def test_closed_form_serialized_batched_hvp(op, dtype, shape, opset, tmp_path):
    x = _value("x", [None] * len(shape), dtype=dtype)
    tape = ir.tape.Tape()
    loss = tape.op(op, [x], {"keepdims": 0})
    source = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    values = (
        (np.arange(np.prod(shape), dtype=np_dtype).reshape(shape) - 2)
        if shape
        else np.array(-0.7, dtype=np_dtype)
    )
    direction = np.full_like(values, 0.3)
    slope = np.sign(values) if op == "ReduceL1" else 2 * values
    curvature_product = np.zeros_like(values) if op == "ReduceL1" else 2 * direction
    gradient = grad(source)
    ir.save(gradient, tmp_path / "gradient.onnx")
    product = jvp(ir.load(tmp_path / "gradient.onnx"))
    executor = runtime_session if opset == 18 else evaluator
    actual = executor(product).run(
        None, {"x": values, product.graph.inputs[-1].name: direction}
    )
    np.testing.assert_allclose(actual[-2], slope)
    np.testing.assert_allclose(actual[-1], curvature_product)
    mapped = vmap(product, in_axes=(0, 0))
    batch = np.stack([values, -values])
    actual = executor(mapped).run(
        None, {"x": batch, mapped.graph.inputs[-1].name: np.stack([direction, direction])}
    )
    np.testing.assert_allclose(actual[-1], np.stack([curvature_product, curvature_product]))
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("keepdims,axes", [(0, None), (1, [1])])
def test_directly_broadcastable_pullback_avoids_axis_reconstruction(keepdims, axes):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    operands = [x]
    if axes is not None:
        operands.append(
            tape.op("Constant", [], {"value": ir.tensor(axes, dtype=ir.DataType.INT64)})
        )
    y = tape.op("ReduceSumSquare", operands, {"keepdims": keepdims})
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    derivative = grad(_model([x], tape, [loss]))
    assert not {"Range", "ScatterElements", "Size"} & {
        node.op_type for node in derivative.graph
    }
    values = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(
        runtime_session(derivative).run(None, {"x": values})[-1], 2 * values
    )


@pytest.mark.parametrize("count", [0, 1, None])
@pytest.mark.parametrize("noop", [0, 1])
def test_axes_length_specialization_preserves_runtime_values(count, noop):
    from onnax import vjp

    x = _value("x", [2, 3])
    axes = _value("axes", [count], dtype=ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op("ReduceSumSquare", [x, axes], {"keepdims": 0, "noop_with_empty_axes": noop})
    # keepdims=0 with unknown axes length has a runtime-dependent rank. A
    # scalar outer loss provides a fixed derivative interface in that case.
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    loss.shape = ir.Shape([])
    source = _model([x, axes], tape, [loss])
    derivative = vjp(source, inputs=["x"])
    nodes = {node.op_type for node in derivative.graph}
    assert ("Range" in nodes) == (count is None and not noop)
    values = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    choices = [[], [0], [-1]] if count is None else ([[]] if count == 0 else [[0], [-1]])
    for selected in choices:
        actual = runtime_session(derivative).run(
            None,
            {
                "x": values,
                "axes": np.array(selected, dtype=np.int64),
                derivative.graph.inputs[-1].name: np.array(1.0, dtype=np.float64),
            },
        )
        np.testing.assert_array_equal(actual[-1], 2 * values)


@pytest.mark.parametrize("op", ["Sum", "Mean"])
@pytest.mark.parametrize("selected", [None, ["c"]])
def test_variadic_broadcast_products_and_hvp(op, selected):
    inputs = [_value("a", [2, 3]), _value("b", [1, 3]), _value("c", [])]
    tape = ir.tape.Tape()
    y = tape.op(op, inputs)
    source = _model(inputs, tape, [y])
    feeds = {
        "a": np.array([[0.2, 0.1, 0.5], [-0.3, 0.7, -0.2]]),
        "b": np.array([[0.4, -0.5, 0.8]]),
        "c": np.array(0.7),
    }
    _products(source, feeds, inputs=selected)
    loss_tape = ir.tape.Tape(source.graph)
    loss = loss_tape.op("ReduceSum", [loss_tape.op("Mul", [y, y])], {"keepdims": 0})
    source.graph.outputs.clear()
    source.graph.outputs.append(loss)
    loss.name = "loss"
    _hvp_check(source, feeds, inputs=selected)


@pytest.mark.parametrize("op", ["Sum", "Mean"])
def test_variadic_repeated_operands_and_batched_gradient(op):
    a, b = _value("a", [2], dtype=ir.DataType.FLOAT), _value("b", [], dtype=ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op(op, [a, a, b])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    derivative = grad(_model([a, b], tape, [loss]))
    mapped = vmap(derivative, in_axes=(0, None))
    actual = runtime_session(mapped).run(
        None,
        {
            "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "b": np.array(0.5, dtype=np.float32),
        },
    )
    scale = 1 if op == "Sum" else 1 / 3
    np.testing.assert_allclose(actual[-2], np.full((2, 2), 2 * scale))
    np.testing.assert_allclose(actual[-1], np.full(2, 2 * scale))


@pytest.mark.parametrize("op", ["Sum", "Mean"])
def test_variadic_first_operand_expands_at_runtime(op):
    a = _value("a", [2, 1], dtype=ir.DataType.FLOAT)
    b = _value("b", [1, 3], dtype=ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op(op, [a, b])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    derivative = grad(_model([a, b], tape, [loss]))
    actual = runtime_session(derivative).run(
        None, {"a": np.ones((2, 1), dtype=np.float32), "b": np.ones((1, 3), dtype=np.float32)}
    )
    scale = 1 if op == "Sum" else 0.5
    np.testing.assert_allclose(actual[-2], np.full((2, 1), 3 * scale))
    np.testing.assert_allclose(actual[-1], np.full((1, 3), 2 * scale))


@pytest.mark.parametrize("op", ["Sum", "Mean"])
@pytest.mark.parametrize("selected", [None, ["a"]])
def test_variadic_pullback_shares_scaling(op, selected):
    inputs = [_value(name, [2, 3]) for name in ("a", "b", "c")]
    tape = ir.tape.Tape()
    source = _model(inputs, tape, [tape.op(op, inputs)])
    derivative = vjp(source, inputs=selected)
    multiplies = [node for node in derivative.graph if node.op_type == "Mul"]
    assert len(multiplies) == (1 if op == "Mean" else 0)
    feeds = {value.name: np.ones((2, 3)) for value in inputs}
    cotangent = np.arange(6.0).reshape(2, 3)
    feeds[derivative.graph.inputs[-1].name] = cotangent
    outputs = evaluator(derivative).run(None, feeds)
    count = len(selected) if selected else len(inputs)
    for output in outputs[-count:]:
        np.testing.assert_allclose(output, cotangent / (3 if op == "Mean" else 1))


def test_sum_pushforward_needs_no_multiplication():
    inputs = [_value(name, [2, 3]) for name in ("a", "b", "c")]
    tape = ir.tape.Tape()
    source = _model(inputs, tape, [tape.op("Sum", inputs)])
    derivative = jvp(source)
    assert all(node.op_type != "Mul" for node in derivative.graph)
    _products(source, {value.name: np.ones((2, 3)) for value in inputs})


@pytest.mark.parametrize("opset", [18, 28])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("shape", [[], [2, 3]])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
def test_log_sum_serialized_batched_hvp(opset, noop, shape, dtype, tmp_path):
    x = _value("x", [None] * len(shape), dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op("ReduceLogSum", [x], {"keepdims": 0, "noop_with_empty_axes": noop})
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    source = _model([x], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    gradient = grad(source)
    ir.save(gradient, tmp_path / "gradient.onnx")
    product = jvp(ir.load(tmp_path / "gradient.onnx"))
    mapped = vmap(product, in_axes=(0, 0))
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    values = np.arange(1, int(np.prod(shape)) + 1, dtype=np_dtype).reshape(shape)
    direction = np.full_like(values, 0.3)
    denominator = values if noop else values.sum()
    slope = np.broadcast_to(1 / denominator, values.shape)
    hvp = np.broadcast_to(
        -(direction if noop else direction.sum()) / denominator**2, values.shape
    )
    executor = runtime_session if opset == 18 else evaluator
    actual = executor(mapped).run(
        None,
        {
            "x": np.stack([values, 2 * values]),
            mapped.graph.inputs[-1].name: np.stack([direction, direction]),
        },
    )
    np.testing.assert_allclose(actual[-2], np.stack([slope, slope / 2]), rtol=2e-6)
    np.testing.assert_allclose(actual[-1], np.stack([hvp, hvp / 4]), rtol=2e-6)
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_log_sum_reuses_primal_with_signed_inputs(scale):
    x = _value("x", [3])
    tape = ir.tape.Tape()
    source = _model([x], tape, [tape.op("ReduceLogSum", [x], {"keepdims": 0})])
    gradient = grad(source)
    assert all(node.op_type != "ReduceSum" for node in gradient.graph)
    product = jvp(gradient)
    values = np.array([-1.0, 2.0, 3.0]) * scale
    direction = np.array([0.2, -0.1, 0.3])
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: direction}
    )
    np.testing.assert_allclose(actual[-2], 1 / values.sum(), rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        actual[-1], -direction.sum() / values.sum() ** 2, rtol=1e-13, atol=0
    )
