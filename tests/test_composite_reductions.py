# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Composite reductions retain axes semantics under differentiation and batching."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import grad, jvp, vmap


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceSumSquare"])
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
    _products(_model([x], tape, [y]), {"x": values})


@pytest.mark.parametrize("op", ["ReduceL1", "ReduceSumSquare"])
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
