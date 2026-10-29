# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Serialized vmap-of-reverse graphs, checked against independent examples."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from onnax import GradPass, VmapPass
from onnax._inference import _infer_shapes


def _session(model, path):
    ir.save(_infer_shapes(model), path)
    proto = onnx.load(path)
    onnx.checker.check_model(proto, full_check=True)
    assert all(not n.domain for n in proto.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _regressor(dtype, opset, stored):
    rng = np.random.default_rng(42)
    parameters = {
        n: rng.normal(scale=0.3, size=s).astype(dtype)
        for n, s in (("w1", (3, 4)), ("b1", (4,)), ("w2", (4, 2)), ("b2", (2,)))
    }
    tensor_dtype = ir.DataType.FLOAT if dtype == np.float32 else ir.DataType.DOUBLE
    x = ir.val("x", dtype=tensor_dtype, shape=["rows", 3])
    target = ir.val("target", dtype=tensor_dtype, shape=["rows", 2])
    tape = ir.tape.Tape()
    p = {
        n: (
            tape.initializer(ir.tensor(v, name=n))
            if stored
            else ir.val(n, dtype=tensor_dtype, shape=v.shape)
        )
        for n, v in parameters.items()
    }
    hidden = tape.op("Tanh", [tape.op("Add", [tape.op("MatMul", [x, p["w1"]]), p["b1"]])])
    prediction = tape.op("Add", [tape.op("MatMul", [hidden, p["w2"]]), p["b2"]])
    prediction.name = "prediction"
    residual = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "loss"
    model = ir.Model(
        ir.Graph(
            [x, target, *([] if stored else p.values())],
            [prediction, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": opset},
            name="regressor",
        ),
        ir_version=10,
    )
    model.metadata_props["test.model"] = "preserved"
    model.graph.metadata_props["test.graph"] = "preserved"
    return model, parameters


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("opset", range(18, 27))
@pytest.mark.parametrize("stored", [False, True])
def test_shared_multi_parameter_gradients_dynamic_batches(dtype, opset, stored, tmp_path):
    source, parameters = _regressor(dtype, opset, stored)
    before = ir.to_proto(source).SerializeToString()
    gradient = GradPass(inputs=list(parameters), output="loss")(source).model
    gradient_before = ir.to_proto(gradient).SerializeToString()
    axes = [0, -1, *([] if stored else [None] * len(parameters))]
    batched = VmapPass(axes, out_axes=-1)(gradient).model
    independent = _session(gradient, tmp_path / "gradient.onnx")
    session = _session(batched, tmp_path / "batched.onnx")
    assert dict(batched.graph.opset_imports) == {"": opset}
    assert batched.metadata_props["test.model"] == "preserved"
    assert batched.graph.metadata_props["test.graph"] == "preserved"
    assert (
        batched.graph.metadata_props["onnax.autodiff.interface"]
        == gradient.graph.metadata_props["onnax.autodiff.interface"]
    )
    assert [v.name for v in batched.graph.inputs] == [v.name for v in source.graph.inputs]
    for name in parameters:
        if stored:
            np.testing.assert_array_equal(
                batched.graph.initializers[name].const_value.numpy(), parameters[name]
            )
        else:
            assert tuple(batched.graph.inputs[2 + list(parameters).index(name)].shape) == (
                parameters[name].shape
            )
    rng = np.random.default_rng(7)
    for batch, rows in ((1, 1), (3, 2), (7, 4)):
        x = rng.normal(size=(batch, rows, 3)).astype(dtype)
        target = rng.normal(size=(batch, rows, 2)).astype(dtype)
        shared = {} if stored else parameters
        actual = session.run(None, {"x": x, "target": np.moveaxis(target, 0, -1), **shared})
        expected = [
            independent.run(None, {"x": a, "target": b, **shared})
            for a, b in zip(x, target, strict=True)
        ]
        for index, value in enumerate(actual):
            reference = np.stack([e[index] for e in expected], axis=-1)
            assert value.shape == reference.shape
            assert value.dtype == dtype
            np.testing.assert_allclose(value, reference, rtol=3e-5, atol=2e-6)
        hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
        prediction = hidden @ parameters["w2"] + parameters["b2"]
        residual = prediction - target
        g = 2 * residual / (rows * 2)
        dh = (g @ parameters["w2"].T) * (1 - hidden**2)
        analytic = [
            np.swapaxes(x, -1, -2) @ dh,
            dh.sum(axis=1),
            np.swapaxes(hidden, -1, -2) @ g,
            g.sum(axis=1),
        ]
        for value, reference in zip(actual[2:], analytic, strict=True):
            np.testing.assert_allclose(
                np.moveaxis(value, -1, 0), reference, rtol=3e-5, atol=2e-6
            )
    assert ir.to_proto(source).SerializeToString() == before
    assert ir.to_proto(gradient).SerializeToString() == gradient_before


def test_parameter_gradients_match_coordinate_differences(tmp_path):
    model, parameters = _regressor(np.float64, 18, False)
    gradient = GradPass(inputs=list(parameters), output="loss")(model).model
    batched = VmapPass([0, 0, None, None, None, None])(gradient).model
    forward = _session(
        VmapPass([0, 0, None, None, None, None])(model).model, tmp_path / "forward.onnx"
    )
    session = _session(batched, tmp_path / "gradient.onnx")
    rng = np.random.default_rng(4)
    feeds = {
        "x": rng.normal(size=(3, 2, 3)),
        "target": rng.normal(size=(3, 2, 2)),
        **parameters,
    }
    actual = session.run(None, feeds)
    epsilon = 1e-5
    for name, value in zip(parameters, actual[2:], strict=True):
        numerical = np.empty_like(value)
        for coordinate in np.ndindex(parameters[name].shape):
            plus, minus = parameters[name].copy(), parameters[name].copy()
            plus[coordinate] += epsilon
            minus[coordinate] -= epsilon
            numerical[(slice(None), *coordinate)] = (
                forward.run(None, {**feeds, name: plus})[1]
                - forward.run(None, {**feeds, name: minus})[1]
            ) / (2 * epsilon)
        np.testing.assert_allclose(value, numerical, rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("noop", [0, 1])
@pytest.mark.parametrize("axis_values", [[], [-1], [0, -1]])
@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean", "ReduceL1", "ReduceLogSum"])
def test_runtime_shared_reduction_controls(op, axis_values, keepdims, noop, tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    axes = ir.val(
        "axes",
        dtype=ir.DataType.INT64,
        shape=[len(axis_values)] if not keepdims else ["count"],
    )
    tape = ir.tape.Tape()
    y = tape.op(op, [x, axes], {"keepdims": keepdims, "noop_with_empty_axes": noop})
    y.name = "y"
    # keepdims=0 has fixed axes cardinality; inferred dimensions can still vary.
    rank = (
        2
        if keepdims or (not axis_values and noop)
        else (0 if not axis_values else 2 - len(axis_values))
    )
    y.shape, y.dtype = ir.Shape([None] * rank), x.dtype
    source = ir.Model(
        ir.Graph([x, axes], [y], nodes=tape.nodes, opset_imports={"": 18}, name="reduction"),
        ir_version=10,
    )
    batched = VmapPass([0, None])(source).model
    single = _session(source, tmp_path / "single.onnx")
    session = _session(batched, tmp_path / "batch.onnx")
    data = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 10 + 0.1
    axis_data = np.array(axis_values, dtype=np.int64)
    actual = session.run(None, {"x": data, "axes": axis_data})[0]
    expected = np.stack([single.run(None, {"x": d, "axes": axis_data})[0] for d in data])
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


@pytest.mark.parametrize("mapped_axes", [True, False])
def test_unrepresentable_runtime_reductions_fail_without_mutation(mapped_axes):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    axes = ir.val("axes", dtype=ir.DataType.INT64, shape=["count"])
    node = ir.node("ReduceSum", [x, axes], {"keepdims": int(mapped_axes)})
    node.outputs[0].name, node.outputs[0].dtype = "y", x.dtype
    model = ir.Model(
        ir.Graph(
            [x, axes],
            node.outputs,
            nodes=[node],
            opset_imports={"": 18},
            name="unrepresentable",
        ),
        ir_version=10,
    )
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ir.passes.PreconditionError, match=r"shared|known axes length"):
        VmapPass([0, 0 if mapped_axes else None])(model)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("shape", [(), (1,), (1, 3), (0, 3)])
def test_unbroadcast_leading_singleton_scalar_and_empty_extents(shape, tmp_path):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=["rows", 3])
    p = ir.val("p", dtype=ir.DataType.DOUBLE, shape=[None] * len(shape))
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, p])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    loss.name = "loss"
    source = ir.Model(
        ir.Graph([x, p], [loss], nodes=tape.nodes, opset_imports={"": 18}, name="broadcast"),
        ir_version=10,
    )
    gradient = GradPass(inputs=["p"])(source).model
    session = _session(VmapPass([0, None])(gradient).model, tmp_path / "broadcast.onnx")
    for batch in (1, 3, 7):
        rows = 0 if shape == (0, 3) else 2
        data = np.arange(batch * rows * 3, dtype=np.float64).reshape(batch, rows, 3)
        actual = session.run(None, {"x": data, "p": np.ones(shape)})[-1]
        expected = data.sum(axis=(1, 2)) if len(shape) < 2 else data.sum(axis=1, keepdims=True)
        if shape == (1,):
            expected = expected[:, None]
        elif shape == (0, 3):
            expected = np.empty((batch, 0, 3))
        assert actual.shape == (batch, *shape)
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("noop", [0, 1])
def test_generated_gradients_reuse_variable_length_shared_axes(noop, tmp_path):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=["rows", 3])
    p = ir.val("p", dtype=ir.DataType.DOUBLE, shape=[3])
    axes = ir.val("axes", dtype=ir.DataType.INT64, shape=["count"])
    tape = ir.tape.Tape()
    reduced = tape.op(
        "ReduceMean",
        [tape.op("Mul", [x, p]), axes],
        {"keepdims": 1, "noop_with_empty_axes": noop},
    )
    # The symbolic engine does not infer rank for unknown reduction-axis values.
    reduced.shape = ir.Shape([None, None])
    loss = tape.op("ReduceSum", [reduced], {"keepdims": 0})
    loss.name = "loss"
    model = ir.Model(
        ir.Graph(
            [x, p, axes],
            [loss],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="runtime_reduction_loss",
        ),
        ir_version=10,
    )
    derivative = GradPass(inputs=["p"])(model).model
    session = _session(
        VmapPass([0, None, None])(derivative).model, tmp_path / "runtime_grad.onnx"
    )
    for batch, rows in ((1, 1), (3, 2), (7, 4)):
        x_data = np.arange(batch * rows * 3, dtype=np.float64).reshape(batch, rows, 3) / 10
        for axis_values in ([], [-1], [0], [0, -1]):
            actual = session.run(
                None,
                {"x": x_data, "p": np.ones(3), "axes": np.array(axis_values, dtype=np.int64)},
            )[-1]
            count = (
                np.prod([x_data.shape[a % 2 + 1] for a in axis_values])
                if axis_values
                else (1 if noop else rows * 3)
            )
            np.testing.assert_allclose(actual, x_data.sum(axis=1) / count)


def test_initializer_default_axes_do_not_specialize_runtime_shapes(tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 4])
    axes = ir.val(
        "axes",
        dtype=ir.DataType.INT64,
        shape=[1],
        const_value=ir.tensor(np.array([0], dtype=np.int64)),
    )
    tape = ir.tape.Tape()
    reduced = tape.op("ReduceSum", [x, axes], {"keepdims": 1})
    reduced.name, reduced.dtype, reduced.shape = "reduced", x.dtype, ir.Shape([None, None])
    shape = tape.op("Shape", [reduced])
    shape.name = "example_shape"
    model = ir.Model(
        ir.Graph(
            [x, axes],
            [reduced, shape],
            nodes=tape.nodes,
            initializers=[axes],
            opset_imports={"": 18},
            name="default_axes",
        ),
        ir_version=10,
    )
    before = ir.to_proto(model).SerializeToString()
    mapped = VmapPass([0, None])(model).model
    assert not isinstance(mapped.graph.outputs[0].shape[1], int)
    assert not isinstance(mapped.graph.outputs[0].shape[2], int)
    session = _session(mapped, tmp_path / "defaults.onnx")
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    for axis in (None, 0, 1, -1):
        controls = {} if axis is None else {"axes": np.array([axis], dtype=np.int64)}
        actual, shapes = session.run(None, {"x": data, **controls})
        expected = data.sum(axis=1 if axis is None else (axis % 2 + 1), keepdims=True)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(shapes, np.tile(expected.shape[1:], (3, 1)))
    np.testing.assert_array_equal(mapped.graph.initializers["axes"].const_value.numpy(), [0])
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceLogSum", "ReduceL1"])
@pytest.mark.parametrize("noop", [0, 1])
def test_runtime_empty_axes_preserve_scalar_example_semantics(op, noop, tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[])
    axes = ir.val("axes", dtype=ir.DataType.INT64, shape=["count"])
    node = ir.node(op, [x, axes], {"keepdims": 1, "noop_with_empty_axes": noop})
    node.outputs[0].name, node.outputs[0].dtype, node.outputs[0].shape = (
        "y",
        x.dtype,
        ir.Shape([]),
    )
    model = ir.Model(
        ir.Graph(
            [x, axes],
            node.outputs,
            nodes=[node],
            opset_imports={"": 18},
            name="scalar_reduction",
        ),
        ir_version=10,
    )
    session = _session(VmapPass([0, None])(model).model, tmp_path / "scalar.onnx")
    data = np.array([0.3, 0.8, 1.2], dtype=np.float32)
    actual = session.run(None, {"x": data, "axes": np.array([], dtype=np.int64)})[0]
    expected = np.log(data) if op == "ReduceLogSum" else data
    np.testing.assert_allclose(actual, expected)


def test_practical_example_saves_and_executes_graphs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "per_example_gradients.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for batch in (1, 3, 7):
        assert f"Batch {batch}: per-example parameter-gradient norms:" in result.stdout
    assert "Norm of mean-loss gradient:" in result.stdout
    for name in (
        "single_example",
        "single_gradient",
        "per_example_gradient",
        "mean_loss_gradient",
    ):
        proto = onnx.load(tmp_path / f"{name}.onnx")
        onnx.checker.check_model(proto, full_check=True)
        text = (tmp_path / f"{name}.txt").read_text(encoding="utf-8")
        assert text in result.stdout
        onnx.checker.check_model(onnx.parser.parse_model(text), full_check=True)
    session = _session(ir.load(tmp_path / "per_example_gradient.onnx"), tmp_path / "ort.onnx")
    mean_session = _session(
        ir.load(tmp_path / "mean_loss_gradient.onnx"), tmp_path / "mean_ort.onnx"
    )
    for batch in (1, 3, 7):
        x = np.arange(batch * 3, dtype=np.float64).reshape(batch, 3) / 10
        target = np.full((batch, 2), 0.7)
        prediction, _, gw, gb = session.run(None, {"features": x, "target": target})
        residual = prediction - target
        np.testing.assert_allclose(gw, x[:, :, None] * residual[:, None, :])
        np.testing.assert_allclose(gb, residual)
        mean_loss, mean_gw, mean_gb = mean_session.run(None, {"features": x, "target": target})
        np.testing.assert_allclose(mean_loss, np.mean(residual**2))
        np.testing.assert_allclose(mean_gw, gw.mean(axis=0))
        np.testing.assert_allclose(mean_gb, gb.mean(axis=0))
