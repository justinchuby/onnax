# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Linear rules emit direct tangent operations without unit multiplications."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import jvp, vjp


@pytest.mark.parametrize("op", ["Add", "Sub"])
@pytest.mark.parametrize("selected", [None, ["a"], ["b"]])
@pytest.mark.parametrize("shapes", [([2, 1], [1, 3]), ([], [2, 3])])
def test_binary_linear_products_and_hvp(op, selected, shapes):
    a, b = [_value(name, shape) for name, shape in zip(("a", "b"), shapes)]
    tape = ir.tape.Tape()
    y = tape.op(op, [a, b])
    source = _model([a, b], tape, [y])
    feeds = {
        value.name: np.asarray(
            np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 0.2
        )
        for value, shape in zip((a, b), shapes)
    }
    for transform in (jvp, vjp):
        derivative = transform(source, inputs=selected)
        assert all(node.op_type != "Mul" for node in derivative.graph)
    _products(source, feeds, inputs=selected)
    loss_tape = ir.tape.Tape(source.graph)
    loss = loss_tape.op("ReduceSum", [loss_tape.op("Mul", [y, y])], {"keepdims": 0})
    source.graph.outputs.clear()
    source.graph.outputs.append(loss)
    loss.name = "loss"
    _hvp_check(source, feeds, inputs=selected)


@pytest.mark.parametrize("op", ["Identity", "Neg"])
@pytest.mark.parametrize("shape", [[], [2, 3]])
def test_unary_linear_tangent_preserves_values_without_multiplication(op, shape):
    x = _value("x", shape)
    tape = ir.tape.Tape()
    source = _model([x], tape, [tape.op(op, [x])])
    values = np.full(shape, 0.7)
    direction = np.full(shape, -0.0)
    for transform in (jvp, vjp):
        derivative = transform(source)
        assert all(node.op_type != "Mul" for node in derivative.graph)
        actual = runtime_session(derivative).run(
            None, {"x": values, derivative.graph.inputs[-1].name: direction}
        )[-1]
        expected = direction if op == "Identity" else -direction
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    _products(source, {"x": values})


@pytest.mark.parametrize("op", ["Add", "Sub"])
def test_repeated_operand_accumulates_linear_contributions(op):
    x = _value("x", ["n"])
    tape = ir.tape.Tape()
    source = _model([x], tape, [tape.op(op, [x, x])])
    values = np.array([0.2, -0.3, 0.7])
    _products(source, {"x": values})
    for transform in (jvp, vjp):
        derivative = transform(source)
        actual = runtime_session(derivative).run(
            None, {"x": values, derivative.graph.inputs[-1].name: np.ones_like(values)}
        )[-1]
        np.testing.assert_array_equal(actual, np.full_like(values, 2 if op == "Add" else 0))
