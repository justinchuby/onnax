# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Numerical and interface regression for example-layer segmented ONNX execution."""

import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnax
from examples.huggingface.segmented import SegmentedSession
from examples.lora.train import session


def nonlinear_model():
    rng = np.random.default_rng(71)
    w = rng.normal(0, 0.2, (3, 4)).astype(np.float32)
    c = rng.normal(0, 0.2, (4, 2)).astype(np.float32)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["h"]),
                helper.make_node("Tanh", ["h"], ["a"]),
                helper.make_node("Add", ["h", "a"], ["skip"]),
                helper.make_node("MatMul", ["skip", "c"], ["prediction"]),
                helper.make_node("Mul", ["prediction", "prediction"], ["square"]),
                helper.make_node("ReduceMean", ["square"], ["loss"], keepdims=0),
            ],
            "nonlinear",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
            [
                helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["batch", 2]),
                helper.make_tensor_value_info("loss", TensorProto.FLOAT, []),
            ],
            initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(c, "c")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    return ir.from_proto(model), w, c


@pytest.mark.parametrize("target_nodes", [1, 3, 1024])
def test_segmented_derivatives_dynamic_batches_and_frozen_source(target_nodes):
    model, w, c = nonlinear_model()
    derivative = onnax.grad(model, inputs=["x"], output="loss")
    original = ir.to_proto(derivative).SerializeToString()
    executor = SegmentedSession(derivative, target_nodes=target_nodes)
    reference = session(original)
    rng = np.random.default_rng(37)
    for batch in (5, 1, 3, 5):
        x = rng.normal(size=(batch, 3)).astype(np.float32)
        actual = executor.run(None, {"x": x})
        expected = reference.run(None, {"x": x})
        for a, b in zip(executor.run([], {"x": x}), expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        h = x @ w
        prediction = (h + np.tanh(h)) @ c
        gradient = ((2 * prediction / prediction.size) @ c.T) * (2 - np.tanh(h) ** 2)
        np.testing.assert_allclose(actual[-1], gradient @ w.T, rtol=1e-5, atol=1e-6)
        names = [executor.outputs[-1], "prediction"]
        selected = executor.run(names, {"x": x})
        np.testing.assert_array_equal(selected[0], actual[-1])
        np.testing.assert_array_equal(selected[1], actual[0])
    assert ir.to_proto(derivative).SerializeToString() == original
    for index in range(len(executor.segments)):
        onnx.checker.check_model(
            f"{executor.directory.name}/part{index}.onnx", full_check=True
        )
    assert executor.last_boundary_bytes > 0
    with pytest.raises(ValueError, match="Missing inputs"):
        executor.run(None, {})
    with pytest.raises(ValueError, match="Unknown inputs"):
        executor.run(None, {"x": x, "unknown": x})
    with pytest.raises(ValueError, match="Unknown outputs"):
        executor.run(["h"], {"x": x})


def test_overridable_initializers_and_direct_outputs():
    def info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [3])

    default = np.array([1, 2, 3], np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["x", "w"], ["y"])],
            "defaults",
            [info("x"), info("w")],
            [info("x"), info("w"), info("y")],
            initializer=[numpy_helper.from_array(default, "w")],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    executor = SegmentedSession(ir.from_proto(model), target_nodes=1)
    x = np.array([4, 5, 6], np.float32)
    for feeds in ({"x": x}, {"x": x, "w": -default}, {"x": x}):
        actual = executor.run(None, feeds)
        expected = session(model.SerializeToString()).run(None, feeds)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    actual[1].fill(100)
    np.testing.assert_array_equal(executor.run(["w"], {"x": x})[0], default)


def test_empty_graph_passthrough():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
    constant = ir.val("c", const_value=ir.tensor(np.ones(3, np.float32)))
    model = ir.Model(
        ir.Graph(
            [x], [x, constant], nodes=[], initializers=[constant], opset_imports={"": 18}
        ),
        ir_version=10,
    )
    executor = SegmentedSession(model)
    values = executor.run(None, {"x": np.arange(3, dtype=np.float32)})
    np.testing.assert_array_equal(values[0], np.arange(3))
    np.testing.assert_array_equal(values[1], np.ones(3))
    assert not executor.segments
    with pytest.raises(TypeError, match="dtype"):
        executor.run(None, {"x": np.ones(3, np.float64)})
    with pytest.raises(ValueError, match="shape"):
        executor.run(None, {"x": np.ones((1, 3), np.float32)})
    with pytest.raises(TypeError, match="NumPy array"):
        executor.run(None, {"x": [0, 1, 2]})


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_segment_limit(limit):
    model, _, _ = nonlinear_model()
    with pytest.raises(ValueError, match="positive integer"):
        SegmentedSession(model, target_nodes=limit)


def test_local_function_and_cross_segment_fanout():
    function = helper.make_function(
        "example",
        "Gain",
        ["x"],
        ["y"],
        [
            helper.make_node(
                "Constant",
                [],
                ["gain"],
                value=numpy_helper.from_array(np.array(3, np.float32)),
            ),
            helper.make_node("Mul", ["x", "gain"], ["y"]),
        ],
        [helper.make_opsetid("", 18)],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Gain", ["x"], ["h"], domain="example"),
                helper.make_node("Add", ["h", "x"], ["y"]),
                helper.make_node("Mul", ["h", "y"], ["z"]),
            ],
            "function_fanout",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", 3])],
        ),
        functions=[function],
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("example", 1)],
        ir_version=10,
    )
    source = ir.from_proto(model)
    original = ir.to_proto(source).SerializeToString()
    executor = SegmentedSession(source, target_nodes=1)
    for batch in (2, 1, 5):
        x = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) / 7
        result = executor.run(None, {"x": x})[0]
        np.testing.assert_allclose(result, 12 * x**2, rtol=1e-6, atol=1e-6)
    assert ir.to_proto(source).SerializeToString() == original
    for index in range(len(executor.segments)):
        onnx.checker.check_model(
            f"{executor.directory.name}/part{index}.onnx", full_check=True
        )


def test_control_flow_requires_unsplit_execution():
    branch = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["result"])],
        "branch",
        [],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "If", ["condition"], ["y"], then_branch=branch, else_branch=branch
                )
            ],
            "control_flow",
            [
                helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    with pytest.raises(ValueError, match="Control-flow"):
        SegmentedSession(ir.from_proto(model))


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX segmentation opt-in"
)
def test_native_segmented_derivatives_across_dynamic_batches():
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as native_session

    library = os.environ.get("ONNAX_MLX_EP_LIBRARY")
    if library is not None:
        register_library(library)
    model, _, _ = nonlinear_model()
    derivative = onnax.grad(model, inputs=["x"], output="loss")
    executor = SegmentedSession(derivative, target_nodes=3, session_factory=native_session)
    reference = session(ir.to_proto(derivative).SerializeToString())
    rng = np.random.default_rng(93)
    for batch in (5, 1, 3, 5):
        feeds = {"x": rng.normal(size=(batch, 3)).astype(np.float32)}
        for actual, expected in zip(
            executor.run(None, feeds), reference.run(None, feeds), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
