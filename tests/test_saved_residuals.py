# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Saved pullbacks must avoid frozen layout copies and preserve overrides."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest

import onnax
from examples.huggingface.residuals import split_vjp

ort = pytest.importorskip("onnxruntime")


def execute(model, feeds):
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return dict(
        zip(
            [value.name for value in model.graph.outputs],
            session.run(None, feeds),
            strict=True,
        )
    )


def linear(*, perm=(1, 0), default_input=False, computed=False, output_weight=False):
    rank3 = perm == (2, 0, 1)
    shape = (3, 4, 1) if rank3 else (5, 4)
    weight = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 20
    w = ir.val("weight", dtype=ir.DataType.FLOAT, shape=shape, const_value=ir.tensor(weight))
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 5, 3] if rank3 else ["batch", 4])
    tape = ir.tape.Tape()
    rhs = w
    initializers = [w]
    inputs = [x]
    if default_input:
        inputs.append(w)
    if computed:
        w.const_value = None
        initializers = []
        inputs.append(w)
        one = tape.op("Constant", [], {"value": ir.tensor(np.float32(1))})
        rhs = tape.op("Add", [w, one])
        rhs.name = "shifted_weight"
    transposed = tape.op("Transpose", [rhs], {"perm": list(perm)} if perm is not None else {})
    transposed.name = "weight_t"
    transposed.dtype = ir.DataType.FLOAT
    transposed.shape = ir.Shape(np.transpose(weight, axes=perm).shape)
    y = tape.op("MatMul", [x, transposed])
    y.name, y.dtype = "prediction", ir.DataType.FLOAT
    y.shape = ir.Shape(["batch", 5, 4] if rank3 else ["batch", 5])
    if output_weight:
        transposed.dtype, transposed.shape = ir.DataType.FLOAT, ir.Shape([4, 5])
    graph = ir.Graph(
        inputs,
        [y, transposed] if output_weight else [y],
        nodes=tape.nodes,
        initializers=initializers,
        opset_imports={"": 18},
        name="linear",
    )
    return ir.Model(graph, ir_version=10), weight


@pytest.mark.parametrize("perm", [(1, 0), None, (2, 0, 1)])
def test_frozen_layout_is_not_a_residual(perm):
    model, weight = linear(perm=perm)
    joint = onnax.vjp(model, inputs=["x"], outputs=["prediction"])
    before = [ir.to_proto(value).SerializeToString() for value in (model, joint)]
    old_f, old_b, old_names = split_vjp(model, joint, optimize_layout=False)
    forward, backward, names = split_vjp(model, joint)
    assert "weight_t" in old_names
    assert not names
    assert "weight" in backward.graph.initializers
    assert [value.name for value in backward.graph.outputs] == [
        value.name for value in old_b.graph.outputs
    ]
    assert before == [ir.to_proto(value).SerializeToString() for value in (model, joint)]
    for batch in (2, 3):
        x = (
            np.arange(batch * (15 if weight.ndim == 3 else 4), dtype=np.float32).reshape(
                (batch, 5, 3) if weight.ndim == 3 else (batch, 4)
            )
            / 10
        )
        primal = execute(forward, {"x": x})
        expected_primal = execute(old_f, {"x": x})
        np.testing.assert_array_equal(primal["prediction"], expected_primal["prediction"])
        seed = np.ones_like(primal["prediction"])
        actual = execute(backward, {"x": x, "prediction_cotangent": seed})
        old = execute(
            old_b,
            {
                "x": x,
                "prediction_cotangent": seed,
                **{name: expected_primal[name] for name in old_names},
            },
        )
        np.testing.assert_allclose(
            actual["x_cotangent"], old["x_cotangent"], rtol=1e-6, atol=1e-6
        )


def test_overridable_frozen_input_and_extra_forward_output():
    model, weight = linear(default_input=True, output_weight=True)
    forward, backward, names = split_vjp(
        model, onnax.vjp(model, inputs=["x"], outputs=["prediction"])
    )
    assert not names
    x = np.ones((2, 4), np.float32)
    override = weight + 2
    primal = execute(forward, {"x": x, "weight": override})
    np.testing.assert_array_equal(primal["weight_t"], override.T)
    seed = np.ones_like(primal["prediction"])
    result = execute(backward, {"x": x, "weight": override, "prediction_cotangent": seed})
    np.testing.assert_allclose(result["x_cotangent"], seed @ override, rtol=1e-6, atol=1e-6)


def test_layout_rewrite_does_not_introduce_a_new_snapshot():
    model, weight = linear(computed=True)
    joint = onnax.vjp(model, inputs=["x"], outputs=["prediction"])
    _, _, old = split_vjp(model, joint, optimize_layout=False)
    forward, backward, names = split_vjp(model, joint)
    assert names == old
    assert "weight_t" in names and "shifted_weight" not in names
    x = np.ones((2, 4), np.float32)
    primal = execute(forward, {"x": x, "weight": weight})
    seed = np.ones_like(primal["prediction"])
    result = execute(
        backward,
        {
            "x": x,
            "weight": weight,
            "prediction_cotangent": seed,
            **{name: primal[name] for name in names},
        },
    )
    np.testing.assert_allclose(
        result["x_cotangent"], seed @ (weight + 1), rtol=1e-6, atol=1e-6
    )


def test_isolated_layout_audit():
    pytest.importorskip("torch")
    pytest.importorskip("mobius")
    from examples.huggingface.residual_audit import run

    report = run(nf4=True, warmup=0, samples=1)
    before, after = (report["variants"][name] for name in ("baseline", "candidate"))
    assert after["residuals"] == before["residuals"] - 1
    assert (
        after["cases"][0]["logical_residual_bytes"]
        < before["cases"][0]["logical_residual_bytes"]
    )
    assert all(case["failed_gradient_tensors"] == 0 for case in report["numerical_checks"])
    assert all(case["logits"]["violations"] == 0 for case in report["numerical_checks"])
    assert all(case["loss"]["violations"] == 0 for case in report["numerical_checks"])
    assert all(value["process_peak_rss_bytes"] > 0 for value in (before, after))
