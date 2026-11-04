# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mixed reverse-over-gradient products retain the transpose and interface."""

import json

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnxscript import optimizer

import onnax


@pytest.mark.parametrize("selected", [None, ["z"]])
def test_nonsymmetric_mixed_reverse_product(selected, tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    z = ir.val("z", dtype=ir.DataType.FLOAT, shape=[3])
    matrix = np.array([[0.7, -0.2, 0.9], [-0.3, 0.8, 0.4]], np.float32)
    tape = ir.tape.Tape()
    weight = tape.initializer(ir.tensor(matrix, name="weight"))
    loss = tape.op(
        "ReduceSum",
        [
            tape.op(
                "Mul", [tape.op("Exp", [x]), tape.op("MatMul", [weight, tape.op("Sin", [z])])]
            )
        ],
        {"keepdims": 0},
    )
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.FLOAT, ir.Shape([])
    source = ir.Model(
        ir.Graph(
            [x, z],
            [loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="mixed_reverse",
        ),
        ir_version=10,
    )
    gradient = onnax.grad(source, inputs=["x"], output="loss")
    before = ir.to_proto(gradient).SerializeToString()
    path = tmp_path / "gradient.onnx"
    ir.save(gradient, path)
    result = onnax.vjp(ir.load(path), inputs=selected)
    onnx.checker.check_model(ir.to_proto(result), full_check=True)
    optimized = optimizer.optimize(result.clone())
    info = json.loads(result.graph.metadata_props["onnax.autodiff.interface"])
    assert info["modes"] == ["grad", "vjp"]
    assert info["selection"]["inputs"] == (["x", "z"] if selected is None else selected)
    assert info["selection"]["outputs"] == [gradient.graph.outputs[-1].name]
    assert len(info["seeds"]) == 1
    point = np.array([0.2, -0.4], np.float32)
    hyper = np.array([0.7, -0.1, 0.4], np.float32)
    for seed in [
        np.zeros(2, np.float32),
        np.ones(2, np.float32),
        np.array([0.4, -0.9], np.float32),
    ]:
        actual = runtime_session(result).run(
            None, {"x": point, "z": hyper, info["seeds"][0]: seed}
        )
        optimized_values = runtime_session(optimized).run(
            None, {"x": point, "z": hyper, info["seeds"][0]: seed}
        )
        for plain, optimized_value in zip(actual, optimized_values, strict=True):
            np.testing.assert_allclose(plain, optimized_value, rtol=1e-4, atol=1e-6)
        gx = np.exp(point) * (matrix @ np.sin(hyper))
        expected = {"x": seed * gx, "z": np.cos(hyper) * (matrix.T @ (np.exp(point) * seed))}
        np.testing.assert_allclose(actual[1], gx, rtol=1e-4, atol=1e-6)
        for value, name in zip(actual[2:], info["selection"]["inputs"], strict=True):
            assert value.shape == expected[name].shape
            np.testing.assert_allclose(value, expected[name], rtol=1e-4, atol=1e-6)
    assert ir.to_proto(gradient).SerializeToString() == before


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_multiple_gradient_outputs_accumulate_seeds(batch, tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    z = ir.val("z", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    tape = ir.tape.Tape()
    product = tape.op("Mul", [x, z])
    loss = tape.op("ReduceSum", [tape.op("Mul", [product, product])], {"keepdims": 0})
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.FLOAT, ir.Shape([])
    source = ir.Model(
        ir.Graph(
            [x, z], [loss], nodes=tape.nodes, opset_imports={"": 18}, name="two_gradients"
        ),
        ir_version=10,
    )
    gradient = onnax.grad(source, inputs=["x", "z"], output="loss")
    reverse = onnax.vjp(gradient)
    path = tmp_path / "reverse.onnx"
    ir.save(reverse, path)
    onnx.checker.check_model(path, full_check=True)
    info = json.loads(reverse.graph.metadata_props["onnax.autodiff.interface"])
    assert info["selection"]["outputs"] == [v.name for v in gradient.graph.outputs[1:]]
    assert len(info["seeds"]) == 2
    rng = np.random.default_rng(5)
    left, right, u, v = (rng.normal(size=(batch, 3)).astype(np.float32) for _ in range(4))
    actual = runtime_session(ir.load(path)).run(
        None, {"x": left, "z": right, **dict(zip(info["seeds"], [u, v], strict=True))}
    )
    expected = [
        np.sum((left * right) ** 2),
        2 * left * right**2,
        2 * right * left**2,
        2 * right**2 * u + 4 * left * right * v,
        4 * left * right * u + 2 * left**2 * v,
    ]
    for value, target in zip(actual, expected, strict=True):
        assert value.shape == target.shape
        np.testing.assert_allclose(value, target, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("selected", [["data"], ["updates"], ["data", "updates"]])
@pytest.mark.parametrize(
    "indices",
    [
        np.array([[0], [0], [-1]], np.int64),
        np.array([[0, 1], [0, 1], [-1, 2]], np.int64),
        np.array([[[0, 1], [1, 0]], [[0, 1], [-1, 2]]], np.int64),
        np.empty((0, 1), np.int64),
    ],
)
def test_additive_scatter_reverse_repeated_negative_and_empty_indices(selected, indices):
    shape = (2, 3, 2)
    update_shape = (*indices.shape[:-1], *shape[indices.shape[-1] :])
    data = ir.val("data", dtype=ir.DataType.FLOAT, shape=shape)
    updates = ir.val("updates", dtype=ir.DataType.FLOAT, shape=update_shape)
    tape = ir.tape.Tape()
    index = tape.initializer(ir.tensor(indices, name="indices"))
    output = tape.op("ScatterND", [data, index, updates], {"reduction": "add"})
    output.name, output.dtype, output.shape = "output", ir.DataType.FLOAT, ir.Shape(shape)
    source = ir.Model(
        ir.Graph(
            [data, updates],
            [output],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="additive_scatter",
        ),
        ir_version=10,
    )
    reverse = onnax.vjp(source, inputs=selected)
    onnx.checker.check_model(ir.to_proto(reverse), full_check=True)
    info = json.loads(reverse.graph.metadata_props["onnax.autodiff.interface"])
    rng = np.random.default_rng(9)
    values = rng.normal(size=shape).astype(np.float32)
    changes = rng.normal(size=update_shape).astype(np.float32)
    seed = rng.normal(size=shape).astype(np.float32)
    actual = runtime_session(reverse).run(
        None, {"data": values, "updates": changes, info["seeds"][0]: seed}
    )
    index_tuple = tuple(np.moveaxis(indices, -1, 0))
    forward = values.copy()
    np.add.at(forward, index_tuple, changes)
    np.testing.assert_allclose(actual[0], forward, rtol=1e-4, atol=1e-6)
    expected = {"data": seed, "updates": seed[index_tuple]}
    for value, name in zip(actual[1:], selected, strict=True):
        assert value.shape == expected[name].shape
        np.testing.assert_array_equal(value, expected[name])


@pytest.mark.parametrize("reduction", ["mul", "max", "min"])
def test_scatter_reverse_rejects_other_reductions(reduction):
    data = ir.val("data", dtype=ir.DataType.FLOAT, shape=[2])
    updates = ir.val("updates", dtype=ir.DataType.FLOAT, shape=[1])
    tape = ir.tape.Tape()
    index = tape.initializer(ir.tensor(np.array([[0]], np.int64), name="index"))
    attributes = {} if reduction is None else {"reduction": reduction}
    output = tape.op("ScatterND", [data, index, updates], attributes)
    output.name, output.dtype, output.shape = "output", ir.DataType.FLOAT, ir.Shape([2])
    source = ir.Model(
        ir.Graph(
            [data, updates],
            [output],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
        ),
        ir_version=10,
    )
    with pytest.raises(NotImplementedError, match="VJP ScatterND requires reduction="):
        onnax.vjp(source)
