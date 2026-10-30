# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""General linear-island factoring, independent of the jelly exporter."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass

from onnax import HoistLinearCoefficientsPass


def _model(dtype=ir.DataType.DOUBLE, shape=(2, 3), mode="sum"):
    v = ir.val("v", dtype=dtype, shape=shape)
    a = ir.val("a", dtype=dtype, shape=(shape[0], 1) if shape else ())
    b = ir.val("b", dtype=dtype, shape=() if mode != "two_inputs" else shape)
    tape = ir.tape.Tape()
    left = tape.op("Mul", [v, a])
    if mode == "chain":
        y = tape.op("Div", [left, b])
    elif mode == "negative":
        y = tape.op("Neg", [tape.op("Neg", [left])])
    elif mode == "nonlinear":
        y = tape.op("Mul", [left, v])
    elif mode == "denominator":
        y = tape.op("Div", [a, left])
    elif mode == "two_inputs":
        y = tape.op("Mul", [tape.op("Add", [left, tape.op("Mul", [b, a])]), a])
    else:
        right = tape.op("Mul", [b, v])
        y = tape.op("Sub" if mode == "sub" else "Add", [left, right])
    y.name = "y"
    y.dtype = dtype
    y.shape = ir.Shape(shape) if shape is not None else None
    return ir.Model(
        ir.Graph([v, a, b], [y], nodes=tape.nodes, opset_imports={"": 18}, name="linear"),
        ir_version=9,
        producer_name="test producer",
    )


def _check(model, inputs=("v",), *, max_terms=64):
    before = ir.to_proto(model).SerializeToString()
    result = HoistLinearCoefficientsPass(inputs=inputs, max_terms=max_terms)(model)
    assert ir.to_proto(model).SerializeToString() == before
    assert result.model is not model
    assert [v.name for v in result.model.graph.inputs] == [v.name for v in model.graph.inputs]
    assert [v.name for v in result.model.graph.outputs] == [
        v.name for v in model.graph.outputs
    ]
    rng = np.random.default_rng(4)
    feeds = {
        v.name: rng.uniform(0.4, 1.2, tuple(v.shape)).astype(v.dtype.numpy())
        for v in model.graph.inputs
    }
    expected = evaluator(model).run(None, feeds)
    for executor in (evaluator(result.model), runtime_session(result.model)):
        for a, b in zip(executor.run(None, feeds), expected):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
    return result


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.DOUBLE])
@pytest.mark.parametrize(
    "mode", ["sum", "sub", "chain", "negative", "nonlinear", "denominator"]
)
def test_products_broadcasts_and_opaque_boundaries(dtype, mode):
    result = _check(_model(dtype, mode=mode))
    assert result.modified == (mode not in ("nonlinear", "denominator"))


def test_multiple_selected_inputs_and_term_limit():
    model = _model(mode="two_inputs")
    assert _check(model, inputs=("v", "b")).modified
    assert not _check(model, inputs=("v", "b"), max_terms=1).modified


@pytest.mark.parametrize("shape", [(), (0, 3)])
def test_scalar_and_empty_tensor(shape):
    _check(_model(shape=shape))


def test_metadata_initializers_extra_outputs_and_noop():
    model = _model()
    graph = model.graph
    graph.metadata_props["purpose"] = "linear-test"
    model.metadata_props["model"] = "preserved"
    graph.outputs[0].metadata_props["output"] = "preserved"
    graph.outputs[0].doc_string = "result docs"
    graph.outputs.append(graph.inputs[0])
    weight = ir.val("unused_weight", const_value=ir.tensor(np.array(3.0)))
    graph.register_initializer(weight)
    result = _check(model)
    assert result.model.producer_name == model.producer_name
    assert dict(result.model.metadata_props) == dict(model.metadata_props)
    assert dict(result.model.graph.metadata_props) == dict(graph.metadata_props)
    assert result.model.graph.outputs[0].metadata_props["output"] == "preserved"
    assert result.model.graph.outputs[0].doc_string == "result docs"
    np.testing.assert_array_equal(
        result.model.graph.initializers["unused_weight"].const_value.numpy(), np.array(3.0)
    )
    assert result.model.graph.outputs[1] is result.model.graph.inputs[0]


def test_unknown_shapes_do_not_prevent_factoring():
    model = _model()
    for value in (*model.graph.inputs, *model.graph.outputs):
        value.shape = None
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert result.modified
    feeds = {"v": np.ones((2, 3)), "a": np.ones((2, 1)), "b": np.array(2.0)}
    np.testing.assert_allclose(
        ReferenceEvaluator(ir.to_proto(result.model)).run(None, feeds)[0],
        ReferenceEvaluator(ir.to_proto(model)).run(None, feeds)[0],
    )


@pytest.mark.parametrize("inputs", [[], "v", ["v", "v"], [1], None])
def test_invalid_selections(inputs):
    with pytest.raises(ValueError, match="nonempty sequence of unique"):
        HoistLinearCoefficientsPass(inputs=inputs)


@pytest.mark.parametrize("max_terms", [0, -1, True, 1.5])
def test_invalid_expansion_limit(max_terms):
    with pytest.raises(ValueError, match="integer >= 1"):
        HoistLinearCoefficientsPass(inputs=["v"], max_terms=max_terms)


def test_unknown_input_and_unsupported_dtype():
    with pytest.raises(ValueError, match=r"unknown input.*available graph inputs"):
        HoistLinearCoefficientsPass(inputs=["missing"])(_model())
    with pytest.raises(TypeError, match="FLOAT or DOUBLE"):
        HoistLinearCoefficientsPass(inputs=["v"])(_model(ir.DataType.INT64))


def test_custom_domain_and_newer_opsets_are_opaque():
    for custom in (True, False):
        model = _model()
        for node in model.graph:
            if custom:
                node.domain = "custom"
            else:
                node.version = 29
        if custom:
            model.graph.opset_imports["custom"] = 1
        before = ir.to_proto(model).SerializeToString()
        result = HoistLinearCoefficientsPass(inputs=["v"])(model)
        assert not result.modified
        assert ir.to_proto(result.model).SerializeToString() == before


def test_deep_coefficient_chain_is_iterative():
    model = _model(mode="chain")
    graph = model.graph
    tape = ir.tape.Tape(graph)
    y = graph.outputs[0]
    for _ in range(1200):
        y = tape.op("Mul", [y, graph.inputs[2]])
        y.dtype = ir.DataType.DOUBLE
    graph.outputs.clear()
    graph.outputs.append(y)
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert result.modified


def test_subgraph_captures_are_conservative_dependencies():
    v = ir.val("v", dtype=ir.DataType.DOUBLE, shape=[2])
    a = ir.val("a", dtype=ir.DataType.DOUBLE, shape=[2])
    condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
    branches = {}
    for name in ("then_branch", "else_branch"):
        branch_tape = ir.tape.Tape()
        output = branch_tape.op("Identity" if name == "then_branch" else "Neg", [v])
        output.dtype, output.shape = ir.DataType.DOUBLE, ir.Shape([2])
        branches[name] = ir.AttrGraph(
            name, ir.Graph([], [output], nodes=branch_tape.nodes, name=name)
        )
    tape = ir.tape.Tape()
    hidden = tape.op("If", [condition], branches)
    hidden.dtype, hidden.shape = ir.DataType.DOUBLE, ir.Shape([2])
    product = tape.op("Mul", [v, hidden])
    y = tape.op("Mul", [product, a])
    y.dtype, y.shape = ir.DataType.DOUBLE, ir.Shape([2])
    model = ir.Model(
        ir.Graph([v, a, condition], [y], nodes=tape.nodes, opset_imports={"": 18}),
        ir_version=9,
    )
    before = ir.to_proto(model).SerializeToString()
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before


def test_multiple_outputs_and_local_functions_are_preserved():
    model = ir.from_onnx_text("""
        <ir_version: 9, opset_import: [ "" : 18, "local" : 1 ]>
        linear (double[2] v, double[2] a) => (double[2] y, double[2] z) {
            left = Mul(v, a)
            right = Mul(a, v)
            y = Add(left, right)
            z = local.square(y)
        }
        <opset_import: [ "" : 18 ], domain: "local">
        square (x) => (y) { y = Mul(x, x) }
    """)
    result = _check(model)
    assert result.modified
    before, after = ir.to_proto(model), ir.to_proto(result.model)
    assert [f.SerializeToString() for f in before.functions] == [
        f.SerializeToString() for f in after.functions
    ]


def test_factored_coefficient_is_independent_of_the_selected_input():
    model = _model()
    factored = HoistLinearCoefficientsPass(inputs=["v"])(model).model
    output = factored.graph.outputs[0].producer()
    assert output.op_type == "Mul"
    assert output.inputs[0] is factored.graph.inputs[0]
    coefficient = output.inputs[1].producer()
    assert coefficient.op_type == "Add"
    assert set(coefficient.inputs) == {factored.graph.inputs[1], factored.graph.inputs[2]}
    assert not HoistLinearCoefficientsPass(inputs=["v"])(factored).modified


def test_distribution_without_invariant_work_is_not_a_rewrite():
    v = ir.val("v", dtype=ir.DataType.DOUBLE, shape=[2])
    w = ir.val("w", dtype=ir.DataType.DOUBLE, shape=[2])
    a = ir.val("a", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    y = tape.op("Mul", [tape.op("Add", [v, w]), a])
    model = ir.Model(
        ir.Graph([v, w, a], [y], nodes=tape.nodes, opset_imports={"": 18}), ir_version=9
    )
    result = HoistLinearCoefficientsPass(inputs=["v", "w"])(model)
    assert not result.modified


@pytest.mark.parametrize("index", [0, 1])
def test_overloaded_arithmetic_is_an_opaque_boundary(index):
    model = _model(mode="chain")
    model.ir_version = 10
    model.graph[index].overload = "custom"
    before = ir.to_proto(model).SerializeToString()
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before


def test_builtin_islands_can_still_factor_around_overloaded_boundaries():
    model = _model(mode="chain")
    model.ir_version = 10
    model.graph[0].overload = "custom"
    boundary = model.graph[0].outputs[0]
    boundary.dtype, boundary.shape = ir.DataType.DOUBLE, ir.Shape([2, 3])
    tape = ir.tape.Tape(model.graph)
    y = tape.op("Mul", [model.graph.outputs[0], model.graph.inputs[1]])
    y.name, y.dtype, y.shape = "result", ir.DataType.DOUBLE, ir.Shape([2, 3])
    model.graph.outputs.clear()
    model.graph.outputs.append(y)
    result = _check(model)
    assert result.modified
    calls = [n for n in result.model.graph if n.overload]
    assert len(calls) == 1
    assert calls[0].op_identifier() == ("", "Mul", "custom")


def _arithmetic_function(name, overload="", domain=""):
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2, 3])
    a = ir.val("a", dtype=ir.DataType.DOUBLE, shape=[2, 3])
    tape = ir.tape.Tape()
    y = tape.op("Sub", [x, a])
    return ir.Function(
        domain,
        name,
        overload,
        graph=ir.Graph(
            [x, a], [y], nodes=tape.nodes, opset_imports={"": 18}, name="difference"
        ),
        attributes=[],
    )


def test_local_arithmetic_overload_preserves_function_semantics():
    model = _model(mode="chain")
    model.ir_version = 10
    function = _arithmetic_function("Mul", "difference")
    model.functions[function.identifier()] = function
    model.graph[0].overload = "difference"
    before = ir.to_proto(model).SerializeToString()
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before
    # Resolve IR function identities explicitly before schema/runtime checks.
    source = InlinePass()(model.clone()).model
    transformed = InlinePass()(result.model.clone()).model
    feeds = {"v": np.full((2, 3), 2.0), "a": np.full((2, 1), 0.5), "b": np.array(4.0)}
    expected = (feeds["v"] - feeds["a"]) / feeds["b"]
    for candidate in (source, transformed):
        np.testing.assert_allclose(evaluator(candidate).run(None, feeds)[0], expected)


@pytest.mark.parametrize("name", ["Add", "Mul", "Constant"])
@pytest.mark.parametrize("domain", ["", "ai.onnx"])
def test_generated_operator_names_cannot_capture_local_functions(name, domain):
    model = _model(mode="negative")
    function = _arithmetic_function(name, domain=domain)
    model.functions[function.identifier()] = function
    before = ir.to_proto(model).SerializeToString()
    result = HoistLinearCoefficientsPass(inputs=["v"])(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before
