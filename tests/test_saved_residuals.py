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
    forward, backward, names = split_vjp(model, joint, optimize_layout=True)
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
        model, onnax.vjp(model, inputs=["x"], outputs=["prediction"]), optimize_layout=True
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
    forward, backward, names = split_vjp(model, joint, optimize_layout=True)
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


@pytest.mark.parametrize("optimize,reuse", [(True, False), (True, True), (False, True)])
def test_bridge_frozen_weight_reuse_is_opt_in(optimize, reuse):
    pytest.importorskip("torch")
    from examples.huggingface.bridge import OnnxModule

    graph, weight = linear()
    graph.graph.inputs[0].shape = ir.Shape([2, 4])
    inputs = np.ones((2, 4), np.float32)
    model = OnnxModule(
        graph,
        {"x": inputs},
        optimize=optimize,
        save_activations=True,
        reuse_frozen_weights=reuse,
    )
    assert ("weight_t" not in model.residual_names) == (optimize and reuse)
    result = model()
    np.testing.assert_allclose(result.detach(), inputs @ weight.T, rtol=1e-4, atol=1e-6)
    result.sum().backward()
    np.testing.assert_allclose(
        model.adapters[0].grad, np.ones((2, 5), np.float32) @ weight, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("optimize_layout", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_scan_body_capture_is_saved_for_the_pullback(optimize_layout, nested):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3, 4])
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=[4])
    tape = ir.tape.Tape()
    coefficient = tape.op("Mul", [a, a])
    coefficient.name, coefficient.dtype, coefficient.shape = (
        "coefficient",
        ir.DataType.FLOAT,
        ir.Shape([4]),
    )
    y = tape.op("Mul", [x, coefficient])
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape([3, 4])
    source = ir.Model(
        ir.Graph([x, a], [y], nodes=tape.nodes, opset_imports={"": 24}, name="scaled_rows"),
        ir_version=12,
    )
    joint = source.clone()
    captured = ir.convenience.create_value_mapping(joint.graph)["coefficient"]
    seed = ir.val("seed", dtype=ir.DataType.FLOAT, shape=[4])
    body = ir.tape.Tape()
    if nested:
        inner_seed = ir.val("inner_seed", dtype=ir.DataType.FLOAT, shape=[4])
        inner = ir.tape.Tape()
        inner_product = inner.op("Mul", [inner_seed, captured])
        inner_product.name, inner_product.dtype, inner_product.shape = (
            "inner_product",
            ir.DataType.FLOAT,
            ir.Shape([4]),
        )
        inner_step = ir.Graph(
            [inner_seed],
            [inner_product],
            nodes=inner.nodes,
            opset_imports={"": 24},
            name="inner_row",
        )
        row_shape = body.initializer(ir.tensor(np.array([1, 4], np.int64), name="row_shape"))
        axes = body.initializer(ir.tensor(np.array([0], np.int64), name="row_axes"))
        row = body.op("Reshape", [seed, row_shape])
        row.dtype, row.shape = ir.DataType.FLOAT, ir.Shape([1, 4])
        result = body.op("Scan", [row], {"body": inner_step, "num_scan_inputs": 1})
        result.dtype, result.shape = ir.DataType.FLOAT, ir.Shape([1, 4])
        product = body.op("Squeeze", [result, axes])
    else:
        product = body.op("Mul", [seed, captured])
    product.name, product.dtype, product.shape = (
        "row_gradient",
        ir.DataType.FLOAT,
        ir.Shape([4]),
    )
    step = ir.Graph(
        [seed],
        [product],
        nodes=body.nodes,
        initializers=body.initializers,
        opset_imports={"": 24},
        name="pullback_row",
    )
    cotangent = ir.val("y_cotangent", dtype=ir.DataType.FLOAT, shape=[3, 4])
    derivative = ir.tape.Tape()
    gradient = derivative.op("Scan", [cotangent], {"body": step, "num_scan_inputs": 1})
    gradient.name, gradient.dtype, gradient.shape = (
        "x_cotangent",
        ir.DataType.FLOAT,
        ir.Shape([3, 4]),
    )
    joint.graph.inputs.append(cotangent)
    joint.graph.extend(derivative.nodes)
    joint.graph.outputs.append(gradient)
    snapshots = [ir.to_proto(model).SerializeToString() for model in (source, joint)]
    forward, backward, names = split_vjp(source, joint, optimize_layout=optimize_layout)
    assert names == ("coefficient",)
    assert [ir.to_proto(model).SerializeToString() for model in (source, joint)] == snapshots
    feeds = {
        "x": np.arange(12, dtype=np.float32).reshape(3, 4),
        "a": np.array([0.5, -1, 2, 0.25], np.float32),
    }
    upstream = np.arange(1, 13, dtype=np.float32).reshape(3, 4) / 10
    primal = execute(forward, feeds)
    expected = execute(joint, {**feeds, "y_cotangent": upstream})
    actual = execute(
        backward, {**feeds, "coefficient": primal["coefficient"], "y_cotangent": upstream}
    )
    np.testing.assert_array_equal(actual["x_cotangent"], upstream * feeds["a"] ** 2)
    np.testing.assert_array_equal(actual["x_cotangent"], expected["x_cotangent"])
