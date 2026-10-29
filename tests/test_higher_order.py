# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from test_autodiff import _evaluator, _model, _value

from onnax import GradPass, JvpPass, VjpPass


def _hvp_check(model, feeds, inputs=None, *, rtol=3e-5, atol=3e-6, epsilon=1e-5):
    """Execute HVPs, finite differences of gradients, and Hessian symmetry."""
    inputs = list(feeds) if inputs is None else inputs
    before = ir.to_proto(model).SerializeToString()
    gradient = GradPass(inputs=inputs)(model).model
    gradient_before = ir.to_proto(gradient).SerializeToString()
    hvp = JvpPass(inputs=inputs)(gradient).model
    info = json.loads(hvp.graph.metadata_props["onnax.autodiff.interface"])
    assert info["modes"] == ["grad", "jvp"]
    assert info["source_inputs"] == [v.name for v in model.graph.inputs]
    assert info["selection"]["outputs"] == [v.name for v in gradient.graph.outputs[1:]]
    assert [v.name for v in hvp.graph.inputs[: len(model.graph.inputs)]] == [
        v.name for v in model.graph.inputs
    ]
    assert [v.name for v in hvp.graph.outputs[: len(gradient.graph.outputs)]] == [
        v.name for v in gradient.graph.outputs
    ]
    rng = np.random.default_rng(42)
    directions = [
        [rng.normal(size=feeds[n].shape).astype(feeds[n].dtype) for n in inputs]
        for _ in range(2)
    ]
    evaluator = _evaluator(hvp)
    ge = _evaluator(gradient)
    products = []
    for direction in directions:
        seed_feeds = {
            v.name: d
            for v, d in zip(hvp.graph.inputs[len(gradient.graph.inputs) :], direction)
        }
        actual = evaluator.run(None, {**feeds, **seed_feeds})
        for a, b in zip(actual[: len(gradient.graph.outputs)], ge.run(None, feeds)):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        plus, minus = dict(feeds), dict(feeds)
        for n, d in zip(inputs, direction):
            plus[n] = feeds[n] + epsilon * d
            minus[n] = feeds[n] - epsilon * d
        gp, gm = ge.run(None, plus)[1:], ge.run(None, minus)[1:]
        hs = actual[len(gradient.graph.outputs) :]
        for h, p, m, n in zip(hs, gp, gm, inputs):
            assert h.shape == feeds[n].shape
            assert h.dtype == feeds[n].dtype
            np.testing.assert_allclose(h, (p - m) / (2 * epsilon), rtol=rtol, atol=atol)
        products.append(hs)
    lhs = sum(np.vdot(u, hv) for u, hv in zip(directions[0], products[1]))
    rhs = sum(np.vdot(v, hu) for v, hu in zip(directions[1], products[0]))
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)
    # Independent mixed-mode identity: H v = grad(J_f v), using coordinate
    # differences of the ordinary forward graph as a numerical oracle.
    # Reverse-over-forward is deliberately not exposed as a supported composition.
    jvp = JvpPass(inputs=inputs)(model).model
    jf = {**feeds}
    for value, d in zip(jvp.graph.inputs[len(model.graph.inputs) :], directions[0]):
        jf[value.name] = d
    je = _evaluator(jvp)
    for n, h in zip(inputs, products[0]):
        expected = np.empty_like(feeds[n])
        for coordinate in np.ndindex(expected.shape):
            plus, minus = dict(jf), dict(jf)
            plus[n], minus[n] = feeds[n].copy(), feeds[n].copy()
            plus[n][coordinate] += epsilon
            minus[n][coordinate] -= epsilon
            expected[coordinate] = (je.run(None, plus)[-1] - je.run(None, minus)[-1]) / (
                2 * epsilon
            )
        np.testing.assert_allclose(h, expected, rtol=rtol, atol=atol)
    assert ir.to_proto(model).SerializeToString() == before
    assert ir.to_proto(gradient).SerializeToString() == gradient_before
    return hvp


@pytest.mark.parametrize(
    "op",
    [
        "Identity",
        "Neg",
        "Exp",
        "Log",
        "Sqrt",
        "Reciprocal",
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Sigmoid",
        "Softplus",
        "Softsign",
        "Erf",
        "Relu",
        "LeakyRelu",
        "Elu",
        "Selu",
        "Softmax",
        "LogSoftmax",
    ],
)
def test_smooth_unary_hvp(op):
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    y = tape.op(op, [x])
    # Nonlinear outer loss makes linear/piecewise-linear inner rules nontrivial.
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    values = np.array([[0.3, 0.7, 1.1], [0.2, 0.4, 0.8]])
    if op not in ("Log", "Sqrt"):
        values[1] *= -1
    options = {"epsilon": 2e-3, "rtol": 2e-3, "atol": 2e-4} if op == "Erf" else {}
    _hvp_check(_model([x], tape, [loss]), {"x": values}, **options)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div", "Pow"])
@pytest.mark.parametrize("shapes", [((), (2, 3)), ((3,), (2, 1, 3)), ((2, 1), (1, 3))])
def test_broadcast_hvp(op, shapes):
    x, z = [_value(n, [None] * len(s)) for n, s in zip(("x", "z"), shapes)]
    tape = ir.tape.Tape()
    y = tape.op(op, [x, z])
    loss = tape.op("ReduceMean", [tape.op("Mul", [y, y])], {"keepdims": 0})
    _hvp_check(
        _model([x, z], tape, [loss]),
        {"x": np.full(shapes[0], 1.3), "z": np.full(shapes[1], 0.7)},
    )


@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 4])
def test_polynomial_hvp_at_zero_and_negative_bases(exponent):
    x = _value("x", [3])
    tape = ir.tape.Tape()
    p = tape.op("Constant", [], {"value": ir.tensor(np.array(exponent, dtype=np.int64))})
    loss = tape.op("ReduceSum", [tape.op("Pow", [x, p])], {"keepdims": 0})
    _hvp_check(_model([x], tape, [loss]), {"x": np.array([0.0, -0.8, 1.2])})


@pytest.mark.parametrize("exponent", [0.0, 1.0, -1.0, 2.0])
def test_active_pow_exponent_mixed_curvature(exponent):
    x, p = _value("x", [2]), _value("p", [])
    tape = ir.tape.Tape()
    loss = tape.op("ReduceSum", [tape.op("Pow", [x, p])], {"keepdims": 0})
    _hvp_check(
        _model([x, p], tape, [loss]),
        {"x": np.array([0.7, 1.3]), "p": np.array(exponent)},
    )


@pytest.mark.parametrize(
    "shapes",
    [
        ((3,), (3,)),
        ((3,), (3, 2)),
        ((2, 3), (3,)),
        ((2, 3), (3, 2)),
        ((3,), (2, 3, 2)),
        ((2, 1, 2, 3), (1, 3, 3, 2)),
    ],
)
def test_matmul_hvp(shapes):
    x, z = [_value(n, [None] * len(s)) for n, s in zip(("x", "z"), shapes)]
    tape = ir.tape.Tape()
    y = tape.op("Tanh", [tape.op("MatMul", [x, z])])
    loss = tape.op("ReduceMean", [y], {"keepdims": 0})
    rng = np.random.default_rng(7)
    _hvp_check(
        _model([x, z], tape, [loss]),
        {"x": rng.normal(size=shapes[0]), "z": rng.normal(size=shapes[1])},
    )


@pytest.mark.parametrize("ta,tb", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_gemm_hvp(ta, tb):
    shapes = [(3, 2) if ta else (2, 3), (2, 3) if tb else (3, 2), ()]
    values = [_value(n, s) for n, s in zip(("a", "b", "c"), shapes)]
    tape = ir.tape.Tape()
    y = tape.op("Gemm", values, {"transA": ta, "transB": tb, "alpha": 0.7, "beta": 1.3})
    loss = tape.op("ReduceSum", [tape.op("Sigmoid", [y])], {"keepdims": 0})
    rng = np.random.default_rng(7)
    _hvp_check(
        _model(values, tape, [loss]),
        {n: rng.normal(size=s) for n, s in zip(("a", "b", "c"), shapes)},
    )


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
@pytest.mark.parametrize("axes", [[], [0], [-1], [0, 2]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
def test_runtime_reduction_hvp(op, axes, keepdims, noop):
    x = _value("x", ["a", "b", "c"])
    axis = _value("axes", ["k"], ir.DataType.INT64)
    tape = ir.tape.Tape()
    y = tape.op(op, [x, axis], {"keepdims": keepdims, "noop_with_empty_axes": noop})
    y.dtype = ir.DataType.DOUBLE
    effective = len(axes) if axes or noop else 3
    y.shape = ir.Shape([None] * (3 if keepdims else 3 - effective))
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    _hvp_check(
        _model([x, axis], tape, [loss]),
        {"x": np.arange(24.0).reshape(2, 3, 4) / 20, "axes": np.array(axes, np.int64)},
        inputs=["x"],
    )


def test_layout_gather_scatter_and_accumulation_hvp():
    x = _value("x", ["n", 3])
    tape = ir.tape.Tape()
    indices = tape.op("Constant", [], {"value": ir.tensor(np.array([1, 1, -1]))})
    y = tape.op("Gather", [x, indices], {"axis": 1})
    y = tape.op("Transpose", [y], {"perm": [1, 0]})
    y = tape.op("Flatten", [y])
    parts = tape.op_multi_out("Split", [y], {"axis": 0, "num_outputs": 3}, num_outputs=3)
    y = tape.op("Concat", [parts[2], parts[0], parts[1]], {"axis": 0})
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    _hvp_check(_model([x], tape, [loss]), {"x": np.arange(6.0).reshape(2, 3) / 10})


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_symbolic_shapes_disconnected_and_analytic_hvp(batch):
    x, bias, unused = _value("x", ["n", 3]), _value("bias", [3]), _value("unused", ["k"])
    tape = ir.tape.Tape()
    y = tape.op("Add", [x, bias])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    model = _model([x, bias, unused], tape, [loss])
    hvp = JvpPass()(GradPass()(model).model).model
    dx, db, du = np.full((batch, 3), 0.7), np.array([0.2, 0.4, 0.6]), np.ones(2)
    feeds = {"x": np.ones((batch, 3)), "bias": np.ones(3), "unused": np.ones(2)}
    for v, d in zip(hvp.graph.inputs[3:], (dx, db, du)):
        feeds[v.name] = d
    actual = _evaluator(hvp).run(None, feeds)
    np.testing.assert_allclose(actual[-3], 2 * (dx + db))
    np.testing.assert_allclose(actual[-2], (2 * (dx + db)).sum(axis=0))
    np.testing.assert_array_equal(actual[-1], np.zeros(2))
    session = runtime_session(hvp)
    for a, b in zip(session.run(None, feeds), actual):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_parameters_serialization_and_seed_name_collisions(tmp_path):
    x = _value("x", [2])
    collision = _value("weight_tangent", [2])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(np.array([0.3, 0.7]), name="weight"))
    y = tape.op("Mul", [x, w])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    model = _model([x, collision], tape, [loss], initializers=tape.initializers)
    gradient = GradPass(inputs=["weight"])(model).model
    ir.save(gradient, tmp_path / "gradient.onnx")
    gradient = ir.load(tmp_path / "gradient.onnx")
    hvp = JvpPass(inputs=["weight"])(gradient).model
    assert hvp.graph.inputs[-1].name == "weight_tangent_1"
    assert "weight" in hvp.graph.initializers
    assert "weight" not in [v.name for v in hvp.graph.inputs]
    info = json.loads(hvp.graph.metadata_props["onnax.autodiff.interface"])
    assert info["seeds"] == ["weight_tangent_1"]
    ir.save(hvp, tmp_path / "hvp.onnx")
    actual = _evaluator(ir.load(tmp_path / "hvp.onnx")).run(
        None,
        {
            "x": np.array([2.0, 3.0]),
            "weight_tangent": np.zeros(2),
            "weight_tangent_1": np.array([0.2, 0.4]),
        },
    )
    np.testing.assert_allclose(actual[-1], 2 * np.array([4.0, 9.0]) * [0.2, 0.4])
    mixed = JvpPass()(gradient).model
    info = json.loads(mixed.graph.metadata_props["onnax.autodiff.interface"])
    assert info["selection"]["inputs"] == ["x", "weight_tangent"]
    actual = _evaluator(mixed).run(
        None,
        {
            "x": np.array([2.0, 3.0]),
            "weight_tangent": np.zeros(2),
            "x_tangent": np.array([0.2, 0.4]),
            "weight_tangent_tangent": np.ones(2),
        },
    )
    np.testing.assert_allclose(actual[-1], 4 * np.array([2.0, 3.0]) * [0.3, 0.7] * [0.2, 0.4])


def test_vjp_seeds_fixed_by_default_and_explicitly_selectable(tmp_path):
    x = _value("x", [2])
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, x])
    model = _model([x], tape, [y])
    pullback = VjpPass()(model).model
    ir.save(pullback, tmp_path / "vjp.onnx")
    pullback = ir.load(tmp_path / "vjp.onnx")
    seed = pullback.graph.inputs[-1].name
    values, u, v = np.array([0.3, 0.7]), np.array([2.0, 3.0]), np.array([0.2, 0.4])
    for inputs in (None, [seed], ["x", seed]):
        hvp = JvpPass(inputs=inputs)(pullback).model
        selected = ["x"] if inputs is None else inputs
        assert len(hvp.graph.inputs) == 2 + len(selected)
        feeds = {"x": values, seed: u}
        for value in hvp.graph.inputs[2:]:
            feeds[value.name] = v
        expected = np.zeros(2)
        if "x" in selected:
            expected += 2 * u * v
        if seed in selected:
            expected += 2 * values * v
        np.testing.assert_allclose(_evaluator(hvp).run(None, feeds)[-1], expected)


def test_multiple_vjp_outputs_and_explicit_output_order():
    x, collision = _value("x", [2]), _value("x_cotangent", [2])
    tape = ir.tape.Tape()
    square = tape.op("Mul", [x, x])
    cube = tape.op("Mul", [square, x])
    source = _model([x, collision], tape, [square, cube])
    reverse = VjpPass(inputs=["x"], outputs=["y1", "y0"])(source).model
    grad_name = reverse.graph.outputs[-1].name
    assert grad_name == "x_cotangent_1"
    hvp = JvpPass(inputs=["x"], outputs=[grad_name, "y0"])(reverse).model
    xval, direction = np.array([0.3, 0.7]), np.array([0.2, 0.4])
    seed_cube, seed_square = np.array([2.0, 3.0]), np.array([-0.5, 0.9])
    feeds = {"x": xval, "x_cotangent": np.zeros(2)}
    for v, seed in zip(reverse.graph.inputs[2:], (seed_cube, seed_square)):
        feeds[v.name] = seed
    feeds[hvp.graph.inputs[-1].name] = direction
    result = _evaluator(hvp).run(None, feeds)
    np.testing.assert_allclose(
        result[-2], (6 * xval * seed_cube + 2 * seed_square) * direction
    )
    np.testing.assert_allclose(result[-1], 2 * xval * direction)
    info = json.loads(hvp.graph.metadata_props["onnax.autodiff.interface"])
    assert info["source_outputs"] == ["y0", "y1"]
    assert info["selection"]["outputs"] == [grad_name, "y0"]


def test_original_integer_interfaces_do_not_become_default_tangents(tmp_path):
    x, axes = _value("x", [2]), _value("axes", [1], ir.DataType.INT64)
    tape = ir.tape.Tape()
    loss = tape.op("ReduceSum", [tape.op("Mul", [x, x]), axes], {"keepdims": 0})
    loss.shape = ir.Shape([])
    gradient = GradPass()(_model([x, axes], tape, [loss])).model
    ir.save(gradient, tmp_path / "gradient.onnx")
    hvp = JvpPass()(ir.load(tmp_path / "gradient.onnx")).model
    info = json.loads(hvp.graph.metadata_props["onnax.autodiff.interface"])
    assert info["source_inputs"] == ["x", "axes"]
    assert info["selection"]["inputs"] == ["x"]
    assert [v.name for v in hvp.graph.inputs] == ["x", "axes", "x_tangent"]
    actual = _evaluator(hvp).run(
        None,
        {
            "x": np.array([0.3, 0.7]),
            "axes": np.array([0], np.int64),
            "x_tangent": np.array([0.2, 0.4]),
        },
    )
    np.testing.assert_allclose(actual[-1], [0.4, 0.8])


@pytest.mark.parametrize("opset", range(18, 27))
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
def test_nonlinear_dynamic_hvp_runtime_and_opsets(opset, dtype, tmp_path):
    x, w = _value("x", ["batch", 2], dtype), _value("w", [2, 1], dtype)
    tape = ir.tape.Tape()
    y = tape.op("Tanh", [tape.op("MatMul", [x, w])])
    loss = tape.op("ReduceMean", [tape.op("Mul", [y, y])], {"keepdims": 0})
    model = _model([x, w], tape, [loss], opset=opset)
    reverse = GradPass(inputs=["w"])(model).model
    hvp = JvpPass(inputs=["w"])(reverse).model
    ir.save(hvp, tmp_path / "hvp.onnx")
    hvp = ir.load(tmp_path / "hvp.onnx")
    session = runtime_session(hvp)
    for batch in (1, 3, 7):
        xval = (np.arange(batch * 2).reshape(batch, 2) / 10).astype(dtype.numpy())
        wval = np.array([[0.3], [0.7]], dtype=dtype.numpy())
        direction = np.array([[0.4], [-0.2]], dtype=dtype.numpy())
        feeds = {"x": xval, "w": wval, hvp.graph.inputs[-1].name: direction}
        t = np.tanh(xval @ wval)
        second = 2 * (1 - t**2) * (1 - 3 * t**2) / batch
        expected = xval.T @ (second * (xval @ direction))
        for result in (session.run(None, feeds), _evaluator(hvp).run(None, feeds)):
            np.testing.assert_allclose(result[-1], expected, rtol=2e-5, atol=1e-7)
            assert result[-1].dtype == dtype.numpy()


def test_reshape_expand_cast_and_where_hvp():
    x = _value("x", [2, 3])
    tape = ir.tape.Tape()
    shape = tape.op("Constant", [], {"value": ir.tensor(np.array([2, 1, 3], np.int64))})
    axes = tape.op("Constant", [], {"value": ir.tensor(np.array([1], np.int64))})
    y = tape.op("Reshape", [x, shape])
    y = tape.op("Expand", [y, shape])
    y = tape.op("Squeeze", [y, axes])
    y = tape.op("Unsqueeze", [y, axes])
    y = tape.op("Cast", [y], {"to": ir.DataType.DOUBLE})
    zero = tape.op("Constant", [], {"value": ir.tensor(np.array(0.0))})
    condition = tape.op("Greater", [y, zero])
    y = tape.op("Where", [condition, tape.op("Mul", [y, y]), tape.op("Sin", [y])])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    _hvp_check(
        _model([x], tape, [loss]),
        {"x": np.array([[-0.3, 0.7, -0.5], [0.4, -0.8, 1.2]])},
    )


@pytest.mark.parametrize("first", [JvpPass, VjpPass, GradPass])
@pytest.mark.parametrize("second", [JvpPass, VjpPass, GradPass])
def test_only_forward_over_reverse_is_authorized(first, second, tmp_path):
    x = _value("x", [])
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op("Exp", [x])])
    derivative = first()(model).model
    ir.save(derivative, tmp_path / "first.onnx")
    reloaded = ir.load(tmp_path / "first.onnx")
    if first is not JvpPass and second is JvpPass:
        hvp = second()(reloaded).model
        ir.save(hvp, tmp_path / "second.onnx")
        for transform in (JvpPass(), VjpPass(), GradPass()):
            with pytest.raises(NotImplementedError, match="Higher-order"):
                transform(ir.load(tmp_path / "second.onnx"))
    else:
        with pytest.raises(NotImplementedError, match="Higher-order"):
            second()(reloaded)


@pytest.mark.parametrize("marker", ["onnax.autodiff", "onnx_ir.autodiff"])
def test_legacy_marker_only_remains_guarded(marker, tmp_path):
    x = _value("x", [])
    tape = ir.tape.Tape()
    model = _model([x], tape, [tape.op("Exp", [x])])
    model.graph.metadata_props[marker] = "grad"
    ir.save(model, tmp_path / "legacy.onnx")
    model = ir.load(tmp_path / "legacy.onnx")
    with pytest.raises(NotImplementedError, match="interface metadata"):
        JvpPass()(model)


@pytest.mark.parametrize("bad", ["null", "{", '{"version": 2}', '{"version": 1}'])
def test_invalid_interface_metadata_remains_guarded(bad):
    x = _value("x", [])
    tape = ir.tape.Tape()
    model = GradPass()(_model([x], tape, [tape.op("Exp", [x])])).model
    model.graph.metadata_props["onnax.autodiff.interface"] = bad
    with pytest.raises(NotImplementedError, match="Higher-order"):
        JvpPass()(model)


def test_unsupported_active_generated_path_fails_precisely():
    x, z = _value("x", [2]), _value("z", [2])
    tape = ir.tape.Tape()
    loss = tape.op("ReduceSum", [tape.op("Abs", [x])], {"keepdims": 0})
    gradient = GradPass(inputs=["x", "z"])(_model([x, z], tape, [loss])).model
    with pytest.raises(NotImplementedError, match="active operator 'Sign'"):
        JvpPass(inputs=["x"])(gradient)
    disconnected = JvpPass(inputs=["z"])(gradient).model
    actual = _evaluator(disconnected).run(
        None, {"x": np.array([0.2, -0.4]), "z": np.ones(2), "z_tangent": np.ones(2)}
    )
    np.testing.assert_array_equal(actual[-2:], np.zeros((2, 2)))


@pytest.mark.parametrize("reduction", [None, "mul", "max", "min"])
def test_scatter_jvp_rejects_nonadditive_updates(reduction):
    x, updates = _value("x", [2]), _value("updates", [2])
    tape = ir.tape.Tape()
    indices = tape.op("Constant", [], {"value": ir.tensor(np.array([[0], [1]], np.int64))})
    attrs = {} if reduction is None else {"reduction": reduction}
    model = _model([x, updates], tape, [tape.op("ScatterND", [x, indices, updates], attrs)])
    with pytest.raises(NotImplementedError, match="reduction='add'"):
        JvpPass()(model)


@pytest.mark.parametrize("inputs", [["x"], ["updates"], ["x", "updates"]])
def test_additive_scatter_jvp_data_updates_and_repeated_indices(inputs):
    x, updates = _value("x", [2]), _value("updates", [2])
    tape = ir.tape.Tape()
    indices = tape.op("Constant", [], {"value": ir.tensor(np.array([[0], [0]], np.int64))})
    y = tape.op("ScatterND", [x, indices, updates], {"reduction": "add"})
    jvp = JvpPass(inputs=inputs)(_model([x, updates], tape, [y])).model
    direction = np.array([0.2, 0.4])
    feeds = {"x": np.array([0.3, 0.7]), "updates": np.array([0.4, 0.6])}
    for seed in jvp.graph.inputs[2:]:
        feeds[seed.name] = direction
    expected = direction.copy() if "x" in inputs else np.zeros(2)
    if "updates" in inputs:
        expected[0] += direction.sum()
    np.testing.assert_allclose(_evaluator(jvp).run(None, feeds)[-1], expected)
