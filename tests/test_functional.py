# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Public signatures, exact graph interfaces, numerical meaning and composition."""

from __future__ import annotations

import inspect
import json
import typing

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from _testing import evaluator
from onnx_ir.passes.common import CheckerPass

import onnax
from onnax import GradPass, JvpPass, VjpPass, VmapPass, grad, jvp, vjp, vmap


def _loss(*, stored=True):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    w = (
        tape.initializer(ir.tensor(np.array([2.0, 3.0]), name="weight"))
        if stored
        else ir.val("weight", dtype=ir.DataType.DOUBLE, shape=[2])
    )
    prediction = tape.op("Mul", [x, w])
    prediction.name = "prediction"
    prediction.type, prediction.shape = ir.TensorType(ir.DataType.DOUBLE), ir.Shape([2])
    residual = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceSum", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "loss"
    loss.type, loss.shape = ir.TensorType(ir.DataType.DOUBLE), ir.Shape([])
    return ir.Model(
        ir.Graph(
            [x, target, *([] if stored else [w])],
            [prediction, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="functional_loss",
        ),
        ir_version=10,
    )


def _names(values):
    return [v.name for v in values]


@pytest.mark.parametrize(
    "function,parameters,defaults",
    [
        (
            vmap,
            ["model", "in_axes", "axis_size", "out_axes"],
            {"axis_size": None, "out_axes": 0},
        ),
        (jvp, ["model", "inputs", "outputs"], {"inputs": None, "outputs": None}),
        (vjp, ["model", "inputs", "outputs"], {"inputs": None, "outputs": None}),
        (grad, ["model", "inputs", "output"], {"inputs": None, "output": None}),
    ],
)
def test_public_signatures_exports_and_documentation(function, parameters, defaults):
    signature = inspect.signature(function)
    assert list(signature.parameters) == parameters
    for name, parameter in signature.parameters.items():
        if name in defaults:
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
            assert parameter.default == defaults[name]
        else:
            assert parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            assert parameter.default == inspect.Parameter.empty
    hints = typing.get_type_hints(function)
    assert hints["model"] is ir.Model and hints["return"] is ir.Model
    assert set(hints) == {*parameters, "return"}
    assert getattr(onnax, function.__name__) is function
    assert function.__name__ in onnax.__all__
    assert "onnx_ir.Model" in inspect.getdoc(function)
    assert "Args:" in inspect.getdoc(function)
    assert set(onnax.__all__) == {
        "vmap",
        "jvp",
        "vjp",
        "grad",
        "VmapPass",
        "HoistLinearCoefficientsPass",
        "JvpPass",
        "VjpPass",
        "GradPass",
    }


@pytest.mark.parametrize(
    "function,transform,kwargs",
    [
        (vmap, VmapPass, {"in_axes": [0, None], "axis_size": 3, "out_axes": [-1, 0]}),
        (jvp, JvpPass, {"inputs": ["weight", "x"], "outputs": ["loss", "prediction"]}),
        (vjp, VjpPass, {"inputs": ["weight", "x"], "outputs": ["loss", "prediction"]}),
        (grad, GradPass, {"inputs": ["weight", "x"], "output": "loss"}),
    ],
)
def test_functional_api_preserves_pass_results_and_source(function, transform, kwargs):
    source = _loss()
    before = ir.to_proto(source).SerializeToString()
    actual = function(source, **kwargs)
    expected = transform(**kwargs)(source)
    assert isinstance(actual, ir.Model) and actual is not source
    assert expected.modified
    # Exact graph equivalence with the compatible pass, including persisted metadata.
    assert (
        ir.to_proto(actual).SerializeToString()
        == ir.to_proto(expected.model).SerializeToString()
    )
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("function", [jvp, vjp])
def test_ordered_selections_determine_exact_seed_and_derivative_interfaces(function):
    source = _loss()
    transformed = function(source, inputs=["weight", "x"], outputs=["loss", "prediction"])
    info = json.loads(transformed.graph.metadata_props["onnax.autodiff.interface"])
    x, target, direction = np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.3, -0.2])
    residual = x * [2, 3] - target
    if function is jvp:
        assert _names(transformed.graph.inputs) == [
            "x",
            "target",
            "weight_tangent",
            "x_tangent",
        ]
        assert _names(transformed.graph.outputs) == [
            "prediction",
            "loss",
            "loss_tangent",
            "prediction_tangent",
        ]
        feeds = {
            "x": x,
            "target": target,
            "weight_tangent": direction,
            "x_tangent": direction * 2,
        }
        dp = x * direction + np.array([2, 3]) * direction * 2
        expected = [np.sum(2 * residual * dp), dp]
    else:
        assert _names(transformed.graph.inputs) == [
            "x",
            "target",
            "loss_cotangent",
            "prediction_cotangent",
        ]
        assert _names(transformed.graph.outputs) == [
            "prediction",
            "loss",
            "weight_cotangent",
            "x_cotangent",
        ]
        feeds = {
            "x": x,
            "target": target,
            "loss_cotangent": np.array(2.0),
            "prediction_cotangent": direction,
        }
        pullback = 4 * residual + direction
        expected = [pullback * x, pullback * [2, 3]]
    actual = evaluator(transformed).run(None, feeds)
    for value, reference in zip(actual[-2:], expected, strict=True):
        np.testing.assert_allclose(value, reference)
    assert info["selection"] == {"inputs": ["weight", "x"], "outputs": ["loss", "prediction"]}
    assert info["seeds"] == _names(transformed.graph.inputs[2:])
    assert info["derivative_outputs"] == _names(transformed.graph.outputs[2:])


def test_defaults_and_scalar_gradient_contract():
    source = _loss()
    forward = jvp(source)
    reverse = vjp(source)
    assert _names(forward.graph.inputs) == ["x", "target", "x_tangent", "target_tangent"]
    assert _names(reverse.graph.outputs) == [
        "prediction",
        "loss",
        "x_cotangent",
        "target_cotangent",
    ]
    with pytest.raises(ValueError, match="rank-zero"):
        grad(source)
    derivative = grad(source, output="loss")
    assert _names(derivative.graph.inputs) == ["x", "target"]
    assert _names(derivative.graph.outputs) == [
        "prediction",
        "loss",
        "x_cotangent",
        "target_cotangent",
    ]
    x, target = np.array([1.0, 2.0]), np.array([0.5, 1.0])
    values = evaluator(derivative).run(None, {"x": x, "target": target})
    residual = x * [2, 3] - target
    np.testing.assert_allclose(values[-2], 2 * residual * [2, 3])
    np.testing.assert_allclose(values[-1], -2 * residual)


@pytest.mark.parametrize("stored", [True, False])
def test_public_per_example_gradient_composition_and_serialized_runtime(stored, tmp_path):
    source = _loss(stored=stored)
    derivative = grad(source, inputs=["weight"], output="loss")
    mapped = vmap(derivative, in_axes=[-1, 0, *([] if stored else [None])], out_axes=-1)
    CheckerPass(full_check=True)(mapped)
    path = tmp_path / "per_example.onnx"
    ir.save(mapped, path)
    CheckerPass(full_check=True)(ir.load(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    independent = evaluator(derivative)
    rng = np.random.default_rng(42)
    for batch in (1, 3, 7):
        x, target = rng.normal(size=(2, batch)), rng.normal(size=(batch, 2))
        shared = {} if stored else {"weight": np.array([2.0, 3.0])}
        actual = session.run(None, {"x": x, "target": target, **shared})
        examples = [
            independent.run(None, {"x": x[:, i], "target": target[i], **shared})
            for i in range(batch)
        ]
        for i, value in enumerate(actual):
            np.testing.assert_allclose(value, np.stack([e[i] for e in examples], axis=-1))
        np.testing.assert_allclose(actual[-1], (2 * (x.T * [2, 3] - target) * x.T).T)


def test_public_hvp_reloaded_graph_and_default_derivative_selection(tmp_path):
    source = _loss()
    derivative = grad(source, inputs=["weight"], output="loss")
    path = tmp_path / "gradient.onnx"
    ir.save(derivative, path)
    reloaded = ir.load(path)
    hvp = jvp(reloaded, inputs=["weight"])
    assert _names(hvp.graph.inputs) == ["x", "target", "weight_tangent"]
    assert _names(hvp.graph.outputs) == [
        "prediction",
        "loss",
        "weight_cotangent",
        "weight_cotangent_tangent",
    ]
    x, direction = np.array([2.0, 3.0]), np.array([0.2, -0.5])
    values = evaluator(hvp).run(
        None, {"x": x, "target": np.zeros(2), "weight_tangent": direction}
    )
    np.testing.assert_allclose(values[-1], 2 * x**2 * direction)
    default = jvp(reloaded)
    assert _names(default.graph.inputs)[-2:] == ["x_tangent", "target_tangent"]
    assert _names(default.graph.outputs)[-1:] == ["weight_cotangent_tangent"]
    for transform in (lambda m: grad(m, output="loss"), vjp, jvp):
        with pytest.raises(NotImplementedError, match="Higher-order"):
            transform(hvp)


def test_seed_names_are_discoverable_after_collisions_and_reload(tmp_path):
    source = _loss()
    source.graph.inputs.append(ir.val("weight_tangent", dtype=ir.DataType.DOUBLE, shape=[]))
    transformed = jvp(source, inputs=["weight"], outputs=["prediction"])
    assert transformed.graph.inputs[-1].name == "weight_tangent_1"
    path = tmp_path / "jvp.onnx"
    ir.save(transformed, path)
    reloaded = ir.load(path)
    info = json.loads(reloaded.graph.metadata_props["onnax.autodiff.interface"])
    assert info["seeds"] == ["weight_tangent_1"]
    assert info["derivative_outputs"] == ["prediction_tangent"]


@pytest.mark.parametrize(
    "transform,error,match",
    [
        (lambda m: jvp(m, inputs="x"), TypeError, "sequence"),
        (lambda m: vjp(m, inputs=["x", "x"]), ValueError, "Duplicate"),
        (lambda m: grad(m, inputs=["missing"], output="loss"), ValueError, "Unknown"),
        (lambda m: jvp(m, outputs=[]), ValueError, "at least one"),
        (lambda m: grad(m, output="prediction"), ValueError, "rank-zero"),
        (lambda m: vmap(m, [None, None]), ir.passes.PreconditionError, "At least one"),
        (lambda m: vmap(m, [0]), ir.passes.PreconditionError, "per graph input"),
        (lambda m: vmap(m, [0, None], axis_size=0), ValueError, "positive"),
        (
            lambda m: vmap(m, [0, None], out_axes=[0]),
            ir.passes.PreconditionError,
            "per graph output",
        ),
    ],
)
def test_functional_errors_preserve_the_source(transform, error, match):
    source = _loss()
    before = ir.to_proto(source).SerializeToString()
    with pytest.raises(error, match=match):
        transform(source)
    assert ir.to_proto(source).SerializeToString() == before


def test_ir_checker_pass_retains_schema_and_full_shape_checks():
    source = _loss()
    source.graph[0].attributes["invalid_attribute"] = ir.AttrInt64("invalid_attribute", 1)
    with pytest.raises(onnx.checker.ValidationError, match="Unrecognized attribute"):
        CheckerPass(full_check=True)(source)
    source = _loss()
    source.graph.outputs[0].shape = ir.Shape([3])
    CheckerPass(full_check=False)(source)
    with pytest.raises(onnx.shape_inference.InferenceError, match="shape"):
        CheckerPass(full_check=True)(source)
