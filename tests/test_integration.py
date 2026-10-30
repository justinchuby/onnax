# SPDX-License-Identifier: MIT
"""Registry integration, serialized execution, and first-order composition."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_shape_inference import infer_symbolic_shapes

from onnax import (
    GradPass,
    JvpPass,
    VjpPass,
    VmapPass,
    _autodiff,
    _batching,
    _rules,
    register_jvp,
    register_vjp,
)


def _square(opset=18):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    tape = ir.tape.Tape()
    y = tape.op("Mul", [x, x])
    y.name, y.dtype, y.shape = "y", x.dtype, ir.Shape([2])
    loss = tape.op("ReduceSum", [y], {"keepdims": 0})
    loss.name, loss.dtype, loss.shape = "loss", x.dtype, ir.Shape([])
    return ir.Model(
        ir.Graph([x], [y, loss], nodes=tape.nodes, opset_imports={"": opset}, name="square"),
        ir_version=10,
    )


def _run(model, feeds, path):
    ir.save(model, path)
    proto = onnx.load(path)
    onnx.checker.check_model(proto, full_check=True)
    session = runtime_session(path)
    return session.run(None, feeds)


@pytest.mark.parametrize("opset", range(18, 27))
@pytest.mark.parametrize("mode", ["batching", "jvp", "vjp", "grad"])
def test_supported_opsets_execute_serialized_graphs(opset, mode, tmp_path):
    model = _square(opset)
    before = ir.to_proto(model).SerializeToString()
    x = np.array([2, 3], dtype=np.float32)
    direction = np.array([4, 5], dtype=np.float32)
    if mode == "batching":
        result = VmapPass([0])(model).model
        x = np.stack([x, x + 1, x + 2])
        feeds = {"x": x}
        expected = [x * x, (x * x).sum(axis=1)]
    elif mode == "grad":
        result = GradPass(inputs=["x"], output="loss")(model).model
        feeds = {"x": x}
        expected = [x * x, (x * x).sum(), 2 * x]
    else:
        transform = JvpPass if mode == "jvp" else VjpPass
        result = transform(inputs=["x"], outputs=["y"])(model).model
        feeds = {"x": x, result.graph.inputs[-1].name: direction}
        expected = [x * x, (x * x).sum(), 2 * x * direction]
    actual = _run(result, feeds, tmp_path / "model.onnx")
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference)
    assert dict(result.graph.opset_imports) == {"": opset}
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("mode", ["batching", "jvp", "vjp", "grad"])
@pytest.mark.parametrize("version", [17, 29])
def test_outside_verified_opset_range_is_rejected(mode, version):
    model = _square(version)
    transforms = {
        "batching": VmapPass([0]),
        "jvp": JvpPass(),
        "vjp": VjpPass(),
        "grad": GradPass(output="loss"),
    }
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises((ValueError, ir.passes.PreconditionError), match="opset"):
        transforms[mode](model)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize(
    "registry_type,transform",
    [
        (_batching._BatchingRegistry, VmapPass([0])),
        (_autodiff._JvpRegistry, JvpPass(outputs=["y"])),
        (_autodiff._VjpRegistry, VjpPass(outputs=["y"])),
        (_autodiff._VjpRegistry, GradPass(output="loss")),
    ],
)
def test_actual_transforms_dispatch_through_typed_registry(
    registry_type, transform, monkeypatch
):
    original = registry_type.lookup
    dispatched = []

    def lookup(self, node, imports):
        dispatched.append((node.domain, node.op_type, dict(imports)))
        return original(self, node, imports)

    monkeypatch.setattr(registry_type, "lookup", lookup)
    transform(_square())
    assert ("", "Mul", {"": 18}) in dispatched


@pytest.mark.parametrize("transform", [VmapPass([0]), JvpPass(), VjpPass()])
def test_explicit_node_version_controls_dispatch(transform):
    model = _square()
    model.graph[0].version = 29
    with pytest.raises((NotImplementedError, ir.passes.PreconditionError), match="opset 29"):
        transform(model)


@pytest.mark.parametrize("transform", [VmapPass([0]), JvpPass(), VjpPass()])
def test_explicit_version_with_a_different_schema_is_rejected(transform):
    model = _square(24)
    node = model.graph[0]
    node.op_type = "Cast"
    node.resize_inputs(1)
    node.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.FLOAT))
    node.version = 18
    with pytest.raises(
        (NotImplementedError, ir.passes.PreconditionError), match="different schema"
    ):
        transform(model)


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_differentiating_a_batched_model(mode, tmp_path):
    model = VmapPass([0])(_square()).model
    transform = JvpPass if mode == "jvp" else VjpPass
    derivative = transform(inputs=["x"], outputs=["loss"])(model).model
    x = np.array([[2, 3], [4, 5]], dtype=np.float32)
    seed = np.array([[1, 2], [3, 4]], dtype=np.float32)
    if mode == "vjp":
        seed = np.array([3, 4], dtype=np.float32)
    feeds = {"x": x, derivative.graph.inputs[-1].name: seed}
    actual = _run(derivative, feeds, tmp_path / "derivative_of_batched.onnx")
    expected = (2 * x * seed).sum(axis=1) if mode == "jvp" else 2 * x * seed[:, None]
    np.testing.assert_allclose(actual[-1], expected)


def test_runtime_unbroadcast_axes_batch_generated_vjp(tmp_path):
    model = VjpPass(inputs=["x"], outputs=["y"])(_square()).model
    before = ir.to_proto(model).SerializeToString()
    batched = VmapPass([0, 0])(model).model
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    seed = np.full_like(x, 2)
    actual = _run(
        batched, {"x": x, batched.graph.inputs[-1].name: seed}, tmp_path / "vmap_vjp.onnx"
    )
    np.testing.assert_allclose(actual[-1], 2 * x * seed)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("mode", ["jvp", "vjp", "grad"])
def test_batching_a_derivative_preserves_composition_contract(mode, tmp_path):
    model = _square()
    model.graph[0].op_type = "Exp"
    model.graph[0].resize_inputs(1)
    if mode == "grad":
        model.graph.inputs[0].shape = ir.Shape([])
        model.graph[0].outputs[0].shape = ir.Shape([])
        model.graph.outputs.pop()
    transforms = {
        "jvp": JvpPass(inputs=["x"], outputs=["y"]),
        "vjp": VjpPass(inputs=["x"], outputs=["y"]),
        "grad": GradPass(inputs=["x"], output="y"),
    }
    derivative = transforms[mode](model).model
    infer_symbolic_shapes(derivative, policy="refine")
    batched = VmapPass([0] * len(derivative.graph.inputs))(derivative).model
    x = np.array([[2, 3], [4, 5]], dtype=np.float32)
    if mode == "grad":
        x = np.array([2, 4], dtype=np.float32)
    feeds = {"x": x}
    direction = np.array([[1, 2], [3, 4]], dtype=np.float32)
    if mode != "grad":
        feeds[batched.graph.inputs[-1].name] = direction
    actual = _run(batched, feeds, tmp_path / "batched_derivative.onnx")
    expected = np.exp(x) if mode == "grad" else np.exp(x) * direction
    np.testing.assert_allclose(actual[-1], expected)
    reloaded = ir.load(tmp_path / "batched_derivative.onnx")
    for transform in (JvpPass(), VjpPass(), GradPass(output="loss")):
        if mode != "jvp" and isinstance(transform, JvpPass):
            second = transform(reloaded).model
            second_feeds = {**feeds, second.graph.inputs[-1].name: np.ones_like(x)}
            np.testing.assert_allclose(
                _run(second, second_feeds, tmp_path / "hvp.onnx")[-1], expected
            )
            continue
        with pytest.raises(NotImplementedError, match="Higher-order"):
            transform(reloaded)


def test_jvp_rule_arity_is_checked(monkeypatch):
    monkeypatch.setattr(_rules, "_EXTENSIONS", _rules._snapshot())
    register_jvp("", "Sign", since_version=18, until_version=28)(lambda *args: [])
    model = _square()
    model.graph[0].op_type = "Sign"
    model.graph[0].resize_inputs(1)
    with pytest.raises(ValueError, match="output arity"):
        JvpPass()(model)


def test_vjp_rule_contribution_keys_are_checked(monkeypatch):
    monkeypatch.setattr(_rules, "_EXTENSIONS", _rules._snapshot())
    register_vjp("", "Sign", since_version=18, until_version=28)(lambda *args: {})
    model = _square()
    model.graph[0].op_type = "Sign"
    model.graph[0].resize_inputs(1)
    with pytest.raises(ValueError, match="input contributions"):
        VjpPass()(model)


def test_autodiff_example_executes_saved_models(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "autodiff.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MSE before/after one gradient step:" in result.stdout
    assert "Directional prediction derivative:" in result.stdout
    assert "Feature cotangent:" in result.stdout
    assert "Hessian-vector product (weight, bias):" in result.stdout
    assert "Directional curvature v.T @ H @ v:" in result.stdout
    for name in ("original", "grad", "jvp", "vjp", "hvp"):
        proto = onnx.load(tmp_path / f"{name}.onnx")
        onnx.checker.check_model(proto, full_check=True)
        text = (tmp_path / f"{name}.txt").read_text(encoding="utf-8")
        assert text in result.stdout
        onnx.checker.check_model(onnx.parser.parse_model(text), full_check=True)
        assert all(not node.domain for node in proto.graph.node)
