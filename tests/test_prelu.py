# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Learnable PReLU slopes, their broadcast pullbacks, and mixed curvature."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import grad, jvp, vjp, vmap


@pytest.mark.parametrize("selection", [["x"], ["slope"], ["x", "slope"]])
@pytest.mark.parametrize(
    "shapes", [((2, 3), ()), ((2, 3), (3,)), ((2, 3, 2), (3, 1)), ((2, 3, 2), (1, 3, 1))]
)
def test_prelu_broadcast_products(selection, shapes):
    x, slope = _value("x", shapes[0]), _value("slope", shapes[1])
    tape = ir.tape.Tape()
    y = tape.op("PRelu", [x, slope])
    values = np.linspace(-1.3, 1.1, np.prod(shapes[0])).reshape(shapes[0])
    _products(
        _model([x, slope], tape, [y]),
        {"x": values, "slope": np.full(shapes[1], 0.3)},
        inputs=selection,
    )


@pytest.mark.parametrize("opset", [18, 24, 28])
def test_prelu_joint_hvp(opset):
    x, slope = _value("x", ["n", 3]), _value("slope", [3])
    tape = ir.tape.Tape()
    y = tape.op("PRelu", [x, slope])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    _hvp_check(
        _model([x, slope], tape, [loss], opset=opset),
        {
            "x": np.array([[-1.2, 0.4, -0.7], [0.2, -0.5, 1.1]]),
            "slope": np.array([0.2, -0.1, 0.0]),
        },
    )


@pytest.mark.parametrize("slope_value", [0.0, 0.2, 1.0, -0.3])
def test_prelu_zero_convention_and_scalar_slope(slope_value):
    x = _value("x", [3], ir.DataType.FLOAT)
    slope = _value("slope", [], ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    y = tape.op("PRelu", [x, slope])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    derivative = grad(_model([x, slope], tape, [loss]))
    values = runtime_session(derivative).run(
        None,
        {
            "x": np.array([-2.0, 0.0, 1.0], np.float32),
            "slope": np.array(slope_value, np.float32),
        },
    )
    np.testing.assert_allclose(values[-2], [slope_value, slope_value, 1.0])
    np.testing.assert_allclose(values[-1], -2.0)
    assert values[-1].shape == ()


def test_prelu_same_graph_handles_dynamic_and_empty_batches():
    x, slope = (
        _value("x", ["n", 3], ir.DataType.FLOAT),
        _value("slope", [3], ir.DataType.FLOAT),
    )
    tape = ir.tape.Tape()
    loss = tape.op("ReduceSum", [tape.op("PRelu", [x, slope])], {"keepdims": 0})
    derivative = grad(_model([x, slope], tape, [loss]), inputs=["slope"])
    session = runtime_session(derivative)
    for batch in (0, 1, 4):
        data = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) - 2
        actual = session.run(None, {"x": data, "slope": np.array([0.1, 0.2, 0.3], np.float32)})
        np.testing.assert_allclose(actual[-1], np.minimum(data, 0).sum(axis=0))


def test_prelu_data_pullback_needs_no_dynamic_broadcast_axis_discovery():
    x = _value("x", ["n", 3], ir.DataType.FLOAT)
    slope = _value("slope", [], ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    model = _model([x, slope], tape, [tape.op("PRelu", [x, slope])])
    pullback = vjp(model, inputs=["x"])
    # Data is never expanded by PReLU. Symbolic dimensions should not introduce
    # a NonZero axis kernel solely to rediscover that equality.
    assert "NonZero" not in {node.op_type for node in pullback.graph}
    session = runtime_session(pullback)
    for batch in (0, 1, 4):
        data = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) - 2
        actual = session.run(
            None,
            {
                "x": data,
                "slope": np.array(0.2, np.float32),
                pullback.graph.inputs[-1].name: np.ones_like(data),
            },
        )[-1]
        np.testing.assert_allclose(actual, np.where(data > 0, 1.0, 0.2))


def test_prelu_stored_parameter_gradients_and_vmap():
    x = _value("x", [3], ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    slope = tape.initializer(ir.tensor(np.array([0.1, 0.2, 0.3], np.float32), name="slope"))
    loss = tape.op("ReduceSum", [tape.op("PRelu", [x, slope])], {"keepdims": 0})
    model = _model([x], tape, [loss], initializers=tape.initializers)
    before = ir.to_proto(model).SerializeToString()
    gradient = grad(model, inputs=["slope"])
    assert [value.name for value in gradient.graph.inputs] == ["x"]
    batched = vmap(gradient, in_axes=(0,))
    data = np.array([[-1.0, 0.0, 2.0], [0.5, -0.7, -1.2]], np.float32)
    np.testing.assert_allclose(
        runtime_session(batched).run(None, {"x": data})[-1], np.minimum(data, 0)
    )
    product = jvp(gradient, inputs=["slope"])
    values = runtime_session(product).run(
        None, {"x": data[1], product.graph.inputs[-1].name: np.ones(3, np.float32)}
    )
    np.testing.assert_array_equal(values[-1], np.zeros(3, np.float32))
    assert ir.to_proto(model).SerializeToString() == before
