# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Concrete broadcast derivatives should not require dynamic axis discovery."""

import numpy as np
import onnx_ir as ir
import pytest
from test_autodiff import _model, _products, _value


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3), (2, 3)),
        ((), ()),
        ((), (2, 3)),
        ((3,), (2, 3)),
        ((2, 1), (1, 3)),
        ((0, 3), (1,)),
        ((2, 0), ()),
    ],
)
def test_concrete_broadcast_products_without_dynamic_axes(shapes):
    x, z = _value("x", shapes[0]), _value("z", shapes[1])
    tape = ir.tape.Tape()
    model = _model([x, z], tape, [tape.op("Mul", [x, z])])
    _, reverse = _products(model, {"x": np.full(shapes[0], 1.3), "z": np.full(shapes[1], 0.7)})
    operators = {node.op_type for node in reverse.graph}
    assert not operators & {"NonZero", "ConstantOfShape", "Shape", "Size"}
    if shapes[0] == shapes[1]:
        assert "ReduceSum" not in operators


@pytest.mark.parametrize(
    "target,source,target_size,source_size,axes,dynamic",
    [
        (["n", 4], ["n", 4], (3, 4), (3, 4), (), False),
        (["n", 4], ["n", 4], (1, 4), (1, 4), (), False),
        (["n", 4], ["n", 4], (0, 4), (0, 4), (), False),
        (["n", 4], ["batch", "n", 4], (3, 4), (2, 3, 4), (0,), False),
        (["n", 1], ["n", "width"], (3, 1), (3, 5), (1,), False),
        (["n", 1], ["n", "width"], (3, 1), (3, 0), (1,), False),
        (["n", 1], ["batch", "n", "width"], (3, 1), (2, 3, 5), (0, 2), False),
        (["n", 1], ["batch", "n", "width"], (3, 1), (0, 3, 5), (0, 2), False),
        ([None, 4], [None, 4], (1, 4), (5, 4), (0,), True),
        (["n", 4], ["m", 4], (1, 4), (5, 4), (0,), True),
        (["n", 4], ["m", 4], (5, 4), (5, 4), (), True),
    ],
)
def test_symbolic_broadcast_proofs_and_runtime_axes(
    target, source, target_size, source_size, axes, dynamic
):
    import onnx
    from _testing import runtime_session

    import onnax

    # The forward result supplies shape information before the new gradient has it.
    gradient = ir.val("gradient", dtype=ir.DataType.FLOAT, shape=[None] * len(source))
    primal = ir.val("primal", dtype=ir.DataType.FLOAT, shape=target)
    forward = ir.val("forward", dtype=ir.DataType.FLOAT, shape=source)
    graph = ir.Graph(
        [gradient, primal], [], nodes=[], opset_imports={"": 18}, name="symbolic_unbroadcast"
    )
    context = onnax.AutodiffContext(graph)
    result = context.unbroadcast(gradient, primal, broadcasted=forward)
    context.output(result, primal, "gradient")
    model = ir.Model(graph, ir_version=10)
    operators = {node.op_type for node in graph}
    assert ("NonZero" in operators) is dynamic
    if not dynamic:
        assert not operators & {"Shape", "Size", "ConstantOfShape"}
        assert ("ReduceSum" in operators) is bool(axes)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    values = np.random.default_rng(7).normal(size=source_size).astype(np.float32)
    actual = runtime_session(model).run(
        None, {"gradient": values, "primal": np.full(target_size, np.nan, np.float32)}
    )[0]
    expected = values.sum(axis=axes, keepdims=True).reshape(target_size)
    assert actual.shape == target_size and actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_symbolic_elementwise_vjp_avoids_shape_programs(rows):
    import onnx
    from _testing import runtime_session

    import onnax

    x, z = (
        _value("x", ["rows", 4], ir.DataType.FLOAT),
        _value("z", ["rows", 4], ir.DataType.FLOAT),
    )
    tape = ir.tape.Tape()
    model = _model([x, z], tape, [tape.op("Mul", [x, z])])
    before = ir.to_proto(model).SerializeToString()
    reverse = onnax.vjp(model)
    assert not {node.op_type for node in reverse.graph} & {"NonZero", "Shape", "Size"}
    onnx.checker.check_model(ir.to_proto(reverse), full_check=True)
    rng = np.random.default_rng(9)
    left, right, seed = (rng.normal(size=(rows, 4)).astype(np.float32) for _ in range(3))
    values = runtime_session(reverse).run(
        None, {"x": left, "z": right, reverse.graph.inputs[-1].name: seed}
    )
    for actual, expected in zip(
        values, (left * right, seed * right, seed * left), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_symbolic_gradient_composes_with_jvp(rows):
    import onnx
    from _testing import runtime_session

    import onnax

    x = _value("x", ["rows", 4], ir.DataType.FLOAT)
    tape = ir.tape.Tape()
    model = _model(
        [x], tape, [tape.op("ReduceSum", [tape.op("Mul", [x, x])], {"keepdims": 0})]
    )
    gradient = onnax.grad(model, inputs=["x"], output="y0")
    product = onnax.jvp(gradient, inputs=["x"], outputs=[gradient.graph.outputs[-1].name])
    assert "NonZero" not in {node.op_type for node in product.graph}
    onnx.checker.check_model(ir.to_proto(product), full_check=True)
    rng = np.random.default_rng(11)
    values, direction = (rng.normal(size=(rows, 4)).astype(np.float32) for _ in range(2))
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: direction}
    )
    np.testing.assert_allclose(actual[1], 2 * values, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[-1], 2 * direction, rtol=1e-6, atol=1e-6)
