# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import grad, jvp, vjp, vmap


@pytest.mark.parametrize("op", ["Max", "Min"])
@pytest.mark.parametrize("selected", [None, ["b"], ["c"]])
def test_extrema_broadcast_products_and_nonlinear_hvp(op, selected):
    inputs = [_value("a", [2, 1]), _value("b", [1, 3]), _value("c", [])]
    tape = ir.tape.Tape()
    y = tape.op(op, inputs)
    source = _model(inputs, tape, [y])
    feeds = {
        "a": np.array([[-2.0], [3.0]]),
        "b": np.array([[-4.0, 0.2, 5.0]]),
        "c": np.array(1.0),
    }
    _products(source, feeds, inputs=selected)
    loss_tape = ir.tape.Tape(source.graph)
    loss = loss_tape.op("ReduceSum", [loss_tape.op("Mul", [y, y])], {"keepdims": 0})
    source.graph.outputs.clear()
    source.graph.outputs.append(loss)
    loss.name = "loss"
    _hvp_check(source, feeds, inputs=selected)


@pytest.mark.parametrize("op", ["Max", "Min"])
@pytest.mark.parametrize("selected", [None, ["a"], ["b"]])
def test_extrema_ties_count_inactive_and_repeated_operands(op, selected):
    a, b = _value("a", [3]), _value("b", [])
    tape = ir.tape.Tape()
    source = _model([a, b], tape, [tape.op(op, [a, a, b])])
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0, 2 / 3, 1]) if op == "Max" else np.array([1, 2 / 3, 0])
    active = selected or ["a", "b"]
    forward = jvp(source, inputs=selected)
    seeds = [np.full(3, 3.0) if name == "a" else np.array(6.0) for name in active]
    feeds = {"a": values, "b": np.array(2.0)}
    feeds.update({value.name: seed for value, seed in zip(forward.graph.inputs[2:], seeds)})
    actual = runtime_session(forward).run(None, feeds)[-1]
    expected = (3 * weights if "a" in active else 0) + (
        6 * (1 - weights) if "b" in active else 0
    )
    np.testing.assert_allclose(actual, expected)
    reverse = vjp(source, inputs=selected)
    actual = runtime_session(reverse).run(
        None, {"a": values, "b": np.array(2.0), reverse.graph.inputs[-1].name: np.ones(3)}
    )
    expected_outputs = {"a": weights, "b": np.array((1 - weights).sum())}
    for name, output in zip(active, actual[-len(active) :]):
        np.testing.assert_allclose(output, expected_outputs[name])


@pytest.mark.parametrize("op", ["Max", "Min"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize("opset", [18, 28])
def test_extrema_serialized_batched_hvp(op, dtype, opset, tmp_path):
    a, b = _value("a", ["n"], dtype=dtype), _value("b", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    y = tape.op(op, [a, b])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    source = _model([a, b], tape, [loss], opset=opset)
    before = ir.to_proto(source).SerializeToString()
    ir.save(grad(source), tmp_path / "gradient.onnx")
    product = jvp(ir.load(tmp_path / "gradient.onnx"), inputs=["a", "b"])
    mapped = vmap(product, in_axes=(0, 0, 0, 0))
    np_dtype = np.float32 if dtype == ir.DataType.FLOAT else np.float64
    av = np.array([[1, 2, 3], [4, 2, 0]], dtype=np_dtype)
    bv = np.full_like(av, 2)
    wa = ((av > bv) if op == "Max" else (av < bv)).astype(np_dtype)
    wa += (av == bv) * 0.5
    wb = 1 - wa
    direction = wa * 0.3 + wb * 0.7
    executor = runtime_session if opset == 18 else evaluator
    actual = executor(mapped).run(
        None,
        dict(
            zip(
                [v.name for v in mapped.graph.inputs],
                [av, bv, np.full_like(av, 0.3), np.full_like(bv, 0.7)],
            )
        ),
    )
    np.testing.assert_allclose(actual[-2], 2 * wa * direction, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual[-1], 2 * wb * direction, rtol=2e-6, atol=1e-7)
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("op", ["Max", "Min"])
@pytest.mark.parametrize("shape", [[], [0, 3], [2, 3]])
def test_single_operand_extrema_are_identity(op, shape):
    x = _value("x", [None] * len(shape))
    tape = ir.tape.Tape()
    source = _model([x], tape, [tape.op(op, [x])])
    for transform in (jvp, vjp):
        derivative = transform(source)
        direction = np.full(shape, 0.7)
        actual = runtime_session(derivative).run(
            None,
            {"x": np.full(shape, -0.3), derivative.graph.inputs[-1].name: direction},
        )[-1]
        np.testing.assert_array_equal(actual, direction)


@pytest.mark.parametrize("op", ["Max", "Min"])
def test_extrema_empty_broadcast_pullback(op):
    a, b = _value("a", [None, 3]), _value("b", [1, 3])
    tape = ir.tape.Tape()
    source = _model([a, b], tape, [tape.op(op, [a, b])])
    derivative = vjp(source)
    actual = runtime_session(derivative).run(
        None,
        {
            "a": np.empty((0, 3)),
            "b": np.ones((1, 3)),
            derivative.graph.inputs[-1].name: np.empty((0, 3)),
        },
    )
    assert actual[-2].shape == (0, 3)
    np.testing.assert_array_equal(actual[-1], np.zeros((1, 3)))
