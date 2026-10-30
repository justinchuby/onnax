# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Global plugin decorators work without arguments on transformation calls."""

import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session

import onnax
from onnax import (
    AutodiffContext,
    BatchingContext,
    JvpPass,
    _rules,
    grad,
    jvp,
    register_batching,
    register_jvp,
    register_vjp,
    vjp,
    vmap,
)


@pytest.fixture(autouse=True)
def isolated_global_registrations(monkeypatch):
    # Only the test harness touches storage; plugin callbacks use public APIs.
    monkeypatch.setattr(_rules, "_EXTENSIONS", _rules._snapshot())


def register_square():
    @register_jvp("demo", "Square", since_version=1, until_version=2)
    def square_jvp(ctx: AutodiffContext, node, tangents):
        x = ctx.input(node)
        return [ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]

    @register_vjp("demo", "Square", since_version=1, until_version=2)
    def square_vjp(ctx: AutodiffContext, node, cotangents, indices):
        x = ctx.input(node)
        return {
            i: ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), cotangents[0])
            for i in indices
        }

    @register_batching("demo", "Square", since_version=1, until_version=2)
    def square_batch(ctx: BatchingContext, node):
        ctx.elementwise(node)

    assert square_jvp.__name__ == "square_jvp"
    assert square_vjp.__name__ == "square_vjp"
    assert square_batch.__name__ == "square_batch"
    return square_jvp


def square_model(scalar=False, version=1):
    formal = ir.val("arg")
    body = ir.tape.Tape()
    result = body.op("Mul", [formal, formal])
    function = ir.Function(
        "demo",
        "Square",
        graph=ir.Graph(
            [formal], [result], nodes=body.nodes, opset_imports={"": 18}, name="square_body"
        ),
        attributes=[],
    )
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    y = tape.op("Square", [x], domain="demo")
    y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([2])
    if scalar:
        y = tape.op("ReduceSum", [y], {"keepdims": 0})
        y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([])
    y.name = "y"
    return ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            opset_imports={"": 18, "demo": version},
            name="extension",
        ),
        ir_version=10,
        functions=[function],
    )


def test_external_autodiff_batching_and_hvp():
    register_square()
    model = square_model()
    before = ir.to_proto(model).SerializeToString()
    data, seed = np.array([2.0, -3.0]), np.array([0.4, 0.7])
    for transform in (jvp, vjp):
        transformed = transform(model)
        feeds = {"x": data, transformed.graph.inputs[-1].name: seed}
        for executor in (evaluator(transformed), runtime_session(transformed)):
            actual = executor.run(None, feeds)
            np.testing.assert_allclose(actual[0], data**2)
            np.testing.assert_allclose(actual[-1], 2 * data * seed)
    batch = np.stack([data, -data])
    mapped = vmap(model, in_axes=(0,))
    np.testing.assert_allclose(runtime_session(mapped).run(None, {"x": batch})[0], batch**2)
    assert ir.to_proto(model).SerializeToString() == before
    loss = square_model(scalar=True)
    derivative = grad(loss)
    product = jvp(derivative)
    np.testing.assert_allclose(
        runtime_session(product).run(None, {"x": data, product.graph.inputs[-1].name: seed})[
            -1
        ],
        2 * seed,
    )
    per_example = vmap(derivative, in_axes=(0,))
    np.testing.assert_allclose(
        runtime_session(per_example).run(None, {"x": batch})[-1], 2 * batch
    )


def test_existing_pass_reads_global_rules_at_invocation():
    transform = JvpPass()
    with pytest.raises(NotImplementedError):
        transform(square_model())
    register_square()
    assert transform(square_model()).model.graph.outputs[-1].name


@pytest.mark.parametrize(
    "decorator,mode",
    [(register_jvp, "jvp"), (register_vjp, "vjp"), (register_batching, "batching")],
)
def test_later_registration_overrides_extensions_and_builtins(decorator, mode):
    register_square()

    def replacement(*args):
        raise RuntimeError("replacement called")

    assert (
        decorator("demo", "Square", since_version=2, until_version=3)(replacement)
        is replacement
    )
    snapshot = _rules._snapshot()
    registry = getattr(snapshot, "_" + mode)
    assert registry.lookup(ir.node("Square", [], domain="demo"), {"demo": 2}) is replacement
    assert (
        registry.lookup(ir.node("Square", [], domain="demo"), {"demo": 1}) is not replacement
    )
    decorator("ai.onnx", "Mul", since_version=18, until_version=18)(replacement)
    source = square_model()
    source.graph.node(0).domain = ""
    source.graph.node(0).op_type = "Mul"
    source.graph.node(0).resize_inputs(2)
    source.graph.node(0).replace_input_with(1, source.graph.inputs[0])
    transform = {"jvp": jvp, "vjp": vjp, "batching": lambda m: vmap(m, in_axes=(0,))}[mode]
    with pytest.raises(RuntimeError, match="replacement called"):
        transform(source)
    decorator("demo", "Square", since_version=1, until_version=3)(lambda *args: None)
    assert registry.lookup(ir.node("Square", [], domain="demo"), {"demo": 2}) is replacement
    assert (
        getattr(_rules._snapshot(), "_" + mode).lookup(
            ir.node("Square", [], domain="demo"), {"demo": 2}
        )
        is not replacement
    )
    # Invalid updates leave both prior registrations and earlier snapshots intact.
    with pytest.raises(ValueError):
        decorator("demo", "Square", since_version=3, until_version=2)(replacement)
    assert registry.lookup(ir.node("Square", [], domain="demo"), {"demo": 2}) is replacement


def test_version_intervals_and_modes_are_independent():
    def callback(ctx, node, tangents):
        x = ctx.input(node)
        return [ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]

    assert (
        register_jvp("demo", "Square", since_version=1, until_version=2)(callback) is callback
    )
    with pytest.raises(NotImplementedError):
        vjp(square_model())
    with pytest.raises(NotImplementedError, match=r"Registered intervals.*1, 2"):
        jvp(square_model(version=3))
    register_jvp("demo", "Square", since_version=3, until_version=3)(callback)
    assert jvp(square_model(version=3)).graph.outputs[-1].name


@pytest.mark.parametrize(
    "mode,decorator",
    [("jvp", register_jvp), ("vjp", register_vjp), ("batching", register_batching)],
)
def test_invalid_rule_results_preserve_source(mode, decorator):
    @decorator("demo", "Square", since_version=1, until_version=1)
    def invalid(*args):
        return None

    model = square_model()
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises((TypeError, ValueError), match=r"rule|register"):
        if mode == "batching":
            vmap(model, in_axes=(0,))
        else:
            {"jvp": jvp, "vjp": vjp}[mode](model)
    assert ir.to_proto(model).SerializeToString() == before


def test_public_protocol_annotations_resolve():
    assert typing.get_type_hints(onnax.JvpRule.__call__)["ctx"] is AutodiffContext
    assert typing.get_type_hints(onnax.VjpRule.__call__)["ctx"] is AutodiffContext
    assert typing.get_type_hints(onnax.BatchingRule.__call__)["ctx"] is BatchingContext


def test_shared_extension_outputs_are_repeated():
    register_square()
    model = square_model()
    model.graph.inputs.append(ir.val("batch", dtype=ir.DataType.DOUBLE, shape=[2]))
    graph = vmap(model, in_axes=(None, 0))
    actual = runtime_session(graph).run(
        None, {"x": np.array([2.0, 3.0]), "batch": np.ones((4, 2))}
    )[0]
    np.testing.assert_allclose(actual, np.tile([4.0, 9.0], (4, 1)))


def test_external_callback_cannot_bypass_random_restrictions():
    called = []

    @register_batching("", "RandomNormal", since_version=18, until_version=28)
    def random_rule(ctx, node):
        called.append(True)

    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
    tape = ir.tape.Tape()
    y = tape.op("RandomNormal", [], {"shape": [2]})
    model = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 18}, name="random"),
        ir_version=10,
    )
    with pytest.raises(ir.passes.PreconditionError, match="random"):
        vmap(model, in_axes=(0,))
    assert not called


@pytest.mark.parametrize("bad", ["foreign", "dtype", "arity", "value"])
def test_invalid_jvp_values_are_rejected(bad):
    @register_jvp("demo", "Square", since_version=1, until_version=1)
    def invalid(ctx, node, tangents):
        if bad == "foreign":
            return [ir.val("foreign", dtype=ir.DataType.DOUBLE, shape=[2])]
        if bad == "dtype":
            v = ctx.op("Identity", tangents[0])
            v.dtype = ir.DataType.FLOAT
            return [v]
        if bad == "arity":
            return []
        return [np.zeros(2)]

    source = square_model()
    before = ir.to_proto(source).SerializeToString()
    with pytest.raises((ValueError, TypeError)):
        jvp(source)
    assert ir.to_proto(source).SerializeToString() == before


def test_registration_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    def derivative(ctx, node, tangents):
        x = ctx.input(node)
        return [ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]

    def install(version):
        register_jvp("demo", "Square", since_version=version, until_version=version)(
            derivative
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(install, range(1, 9)))
    for version in range(1, 9):
        assert jvp(square_model(version=version)).graph.outputs[-1].name


def test_external_rule_can_add_a_standard_operator():
    @register_jvp("ai.onnx", "Sign", since_version=18, until_version=28)
    def sign_jvp(ctx, node, tangents):
        return [ctx.like(ctx.input(node))]

    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    y = tape.op("Sign", [x])
    source = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 18}, name="sign"),
        ir_version=10,
    )
    graph = jvp(source)
    np.testing.assert_array_equal(
        runtime_session(graph).run(
            None, {"x": np.array([2.0, -3.0]), graph.graph.inputs[-1].name: np.ones(2)}
        )[-1],
        np.zeros(2),
    )


def test_custom_rule_example(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "custom_rules.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(tmp_path.glob("*.onnx"))) == 6


def test_multi_output_batching_and_autodiff():
    @register_jvp("demo", "Pair", since_version=1, until_version=1)
    def pair_jvp(ctx, node, tangents):
        x = ctx.input(node)
        return [tangents[0], ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]

    @register_batching("demo", "Pair", since_version=1, until_version=1)
    def pair_batch(ctx, node):
        inputs, mapped = ctx.prepare(node)
        ctx.register_outputs(node, rank=ctx.rank(inputs[0]), mapped=mapped)

    formal = ir.val("arg")
    body = ir.tape.Tape()
    identity = body.op("Identity", [formal])
    squared = body.op("Mul", [formal, formal])
    function = ir.Function(
        "demo",
        "Pair",
        attributes=[],
        graph=ir.Graph(
            [formal],
            [identity, squared],
            nodes=body.nodes,
            opset_imports={"": 18},
            name="pair_body",
        ),
    )
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    pair = ir.node("Pair", [x], domain="demo", num_outputs=2)
    for i, value in enumerate(pair.outputs):
        value.name, value.dtype, value.shape = f"y{i}", ir.DataType.DOUBLE, ir.Shape([2])
    model = ir.Model(
        ir.Graph(
            [x], pair.outputs, nodes=[pair], opset_imports={"": 18, "demo": 1}, name="pair"
        ),
        ir_version=10,
        functions=[function],
    )
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batched = vmap(model, in_axes=(0,))
    actual = runtime_session(batched).run(None, {"x": data})
    np.testing.assert_allclose(actual[0], data)
    np.testing.assert_allclose(actual[1], data**2)
    derivative = jvp(model)
    actual = runtime_session(derivative).run(
        None, {"x": data[0], derivative.graph.inputs[-1].name: data[1]}
    )
    np.testing.assert_allclose(actual[-2], data[1])
    np.testing.assert_allclose(actual[-1], 2 * data[0] * data[1])
