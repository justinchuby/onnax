# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Explicit products own deterministic recurrent function implementations."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import CheckerPass

import onnax
from onnax import _rules

DOMAIN = "onnax.tests.recurrent"


@pytest.fixture(autouse=True)
def isolated_rules(monkeypatch):
    monkeypatch.setattr(_rules, "_EXTENSIONS", _rules._snapshot())


def model(*, nested=False, random=False):
    running = ir.val("step_state", dtype=ir.DataType.FLOAT, shape=[2, 3])
    item = ir.val("step_item", dtype=ir.DataType.FLOAT, shape=[2, 3])
    body = ir.tape.Tape()
    value = body.op("Add", [running, item])
    if random:
        value = body.op(
            "Add",
            [
                value,
                body.op(
                    "RandomNormalLike", [item], domain="ai.onnx" if random == "ai.onnx" else ""
                ),
            ],
        )
    scan_body = ir.Graph(
        [running, item],
        [value, body.op("Identity", [value])],
        nodes=body.nodes,
        opset_imports={"": 18},
        name="scan_step",
    )
    argument = ir.val("sequence", dtype=ir.DataType.FLOAT, shape=[7, 2, 3])
    tape = ir.tape.Tape()
    shape = tape.op("Shape", [argument], {"start": 1})
    initial = tape.op(
        "ConstantOfShape", [shape], {"value": ir.tensor(np.asarray([0], np.float32))}
    )
    outputs = tape.op_multi_out(
        "Scan", [initial, argument], {"num_scan_inputs": 1, "body": scan_body}, num_outputs=2
    )
    function = ir.Function(
        DOMAIN,
        "Core" if nested else "Boundary",
        attributes=[],
        graph=ir.Graph(
            [argument], outputs, nodes=tape.nodes, opset_imports={"": 18}, name="scan_function"
        ),
    )
    functions = [function]
    if nested:
        formal = ir.val("outer_sequence", dtype=ir.DataType.FLOAT, shape=[7, 2, 3])
        wrapper = ir.tape.Tape()
        values = wrapper.op_multi_out("Core", [formal], domain=DOMAIN, num_outputs=2)
        functions.append(
            ir.Function(
                DOMAIN,
                "Boundary",
                attributes=[],
                graph=ir.Graph(
                    [formal],
                    values,
                    nodes=wrapper.nodes,
                    opset_imports={"": 18, DOMAIN: 1},
                    name="wrapper",
                ),
            )
        )
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[7, 2, 3])
    root = ir.tape.Tape()
    results = root.op_multi_out("Boundary", [x], domain=DOMAIN, num_outputs=2)
    for value, name, shape in zip(
        results, ("total", "prefix"), ([2, 3], [7, 2, 3]), strict=True
    ):
        value.name, value.dtype, value.shape = name, ir.DataType.FLOAT, ir.Shape(shape)
    result = ir.Model(
        ir.Graph(
            [x],
            results,
            nodes=root.nodes,
            opset_imports={"": 18, DOMAIN: 1},
            name="function_boundary",
        ),
        ir_version=10,
        functions=functions,
    )
    if random == "ai.onnx":
        for graph in (result.graph, scan_body, *(f.graph for f in functions)):
            graph.opset_imports["ai.onnx"] = 18
    return result


def register(*, reverse_only=False):
    @onnax.register_vjp(DOMAIN, "Boundary", since_version=1, until_version=1)
    def reverse(ctx, node, cotangents, indices):
        gradient = ctx.op("CumSum", cotangents[1], ctx.constant(0), reverse=1)
        final = ctx.op("Expand", cotangents[0], ctx.op("Shape", ctx.input(node)))
        return {0: ctx.op("Add", gradient, final)}

    if reverse_only:
        return

    @onnax.register_jvp(DOMAIN, "Boundary", since_version=1, until_version=1)
    def forward(ctx, node, tangents):
        dx = tangents[0]
        total = ctx.op("ReduceSum", dx, ctx.constant([0]), keepdims=0)
        prefix = ctx.op("CumSum", dx, ctx.constant(0))
        return (total, prefix)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", [onnax.jvp, onnax.vjp])
def test_registered_scan_function_products(nested, mode):
    register()
    source = model(nested=nested)
    before = ir.to_proto(source).SerializeToString()
    product = mode(source)
    CheckerPass(full_check=True)(product)
    rng = np.random.default_rng(71)
    x = rng.normal(size=(7, 2, 3)).astype(np.float32)
    extras = [
        rng.normal(size=tuple(value.shape)).astype(np.float32)
        for value in list(product.graph.inputs)[1:]
    ]
    feeds = {
        "x": x,
        **{
            value.name: array
            for value, array in zip(list(product.graph.inputs)[1:], extras, strict=True)
        },
    }
    output = runtime_session(product).run(None, feeds)
    np.testing.assert_allclose(output[0], np.cumsum(x, axis=0)[-1], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(output[1], np.cumsum(x, axis=0), rtol=1e-4, atol=1e-6)
    if mode is onnax.jvp:
        expected = np.cumsum(extras[0], axis=0)
        np.testing.assert_allclose(output[2], expected[-1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(output[3], expected, rtol=1e-4, atol=1e-6)
    else:
        expected = np.flip(np.cumsum(np.flip(extras[1], axis=0), axis=0), axis=0) + extras[0]
        np.testing.assert_allclose(output[2], expected, rtol=1e-4, atol=1e-6)
    assert ir.to_proto(source).SerializeToString() == before


def test_unregistered_scan_body_is_rejected():
    with pytest.raises(NotImplementedError, match="control flow"):
        onnax.vjp(model())


def test_boundary_registration_is_mode_specific():
    register(reverse_only=True)
    onnax.vjp(model())
    with pytest.raises(NotImplementedError, match="control flow"):
        onnax.jvp(model())


@pytest.mark.parametrize("random", [True, "ai.onnx"])
def test_nested_randomness_remains_rejected(random):
    register()
    with pytest.raises(NotImplementedError, match="randomness"):
        onnax.vjp(model(nested=True, random=random))


def test_top_level_control_flow_remains_rejected():
    register()
    source = model()
    function = next(iter(source.functions.values()))
    direct = ir.Model(function.graph.clone(deep_copy=True), ir_version=10)
    with pytest.raises(NotImplementedError, match="control flow"):
        onnax.vjp(direct)


def test_graph_attribute_references_require_binding():
    register()
    source = model()
    function = next(iter(source.functions.values()))
    scan = next(node for node in function.graph if node.op_type == "Scan")
    body = scan.attributes["body"].as_graph()
    function.attributes["loop_body"] = ir.Attr("loop_body", ir.AttributeType.GRAPH, body)
    scan.attributes["body"] = ir.Attr(
        "body", ir.AttributeType.GRAPH, None, ref_attr_name="loop_body"
    )
    with pytest.raises(NotImplementedError, match="graph-valued"):
        onnax.vjp(source)
