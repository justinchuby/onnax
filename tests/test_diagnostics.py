# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Errors identify the failed contract and a semantics-safe next action."""

from __future__ import annotations

import json

import numpy as np
import onnx_ir as ir
import pytest
from onnx_shape_inference import ShapeInferenceError

from onnax import VmapPass, _inference, _rule_registry, grad, jvp, vjp, vmap
from onnax._batching import _transform as batching


def _model(op="Identity", shapes=((),), *, attrs=None, opset=18, dtype=ir.DataType.FLOAT):
    inputs = [ir.val(f"x{i}", dtype=dtype, shape=shape) for i, shape in enumerate(shapes)]
    node = ir.node(op, inputs, attrs)
    node.name = "offending_node"
    node.outputs[0].name = "result"
    return ir.Model(
        ir.Graph(
            inputs, node.outputs, nodes=[node], opset_imports={"": opset}, name="diagnostic"
        ),
        ir_version=10,
    )


def _contains(error, *fragments):
    text = str(error)
    for fragment in fragments:
        assert fragment in text, text


@pytest.mark.parametrize("size", [0, -2, True, 2.5])
def test_invalid_batch_size_reports_value_and_dynamic_alternative(size):
    with pytest.raises(ValueError) as caught:
        vmap(_model(), [0], axis_size=size)
    _contains(caught.value, repr(size), "positive integer", "None", "mapped input")


@pytest.mark.parametrize(
    "kwargs,fragments",
    [
        ({"in_axes": []}, ("in_axes=()", "expected 1", "x0", "graph-input order")),
        ({"in_axes": [None]}, ("in_axes=(None,)", "At least one", "original model")),
        (
            {"in_axes": [0], "out_axes": []},
            ("out_axes=()", "expected 1", "result", "one integer"),
        ),
        ({"in_axes": [2]}, ("Axis 2", "in_axes", "x0", "rank 1", "[-1, 0]")),
        (
            {"in_axes": [0], "out_axes": True},
            ("Axis True", "out_axes[0]", "result", "not bool"),
        ),
    ],
)
def test_batch_interface_errors_report_actual_selection_and_remedy(kwargs, fragments):
    source = _model()
    before = ir.to_proto(source).SerializeToString()
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, **kwargs)
    _contains(caught.value, *fragments)
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("argument", ["in_axes", "out_axes"])
def test_noniterable_axes_preserve_type_error_and_cause(argument):
    with pytest.raises(TypeError) as caught:
        VmapPass(
            **({"in_axes": 7} if argument == "in_axes" else {"in_axes": [0], "out_axes": None})
        )
    _contains(caught.value, argument, "sequence", "graph")
    assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("transform", [jvp, vjp, grad])
@pytest.mark.parametrize(
    "selection,exception,fragments",
    [
        (["typo"], ValueError, ("Unknown inputs", "typo", "Available inputs", "x0", "exact")),
        (["x0", "x0"], ValueError, ("Duplicate", "x0", "once", "Available inputs")),
        ([], ValueError, ("selection=[]", "Eligible floating", "x0", "nonempty")),
        ("x0", TypeError, ("sequence", "'x0'", "list", "Available inputs")),
        ([["x0"]], TypeError, ("non-string", "Available inputs", "x0")),
    ],
)
def test_autodiff_selections_report_choices(transform, selection, exception, fragments):
    with pytest.raises(exception) as caught:
        transform(_model(), inputs=selection)
    _contains(caught.value, *fragments)


@pytest.mark.parametrize("transform", [jvp, vjp])
def test_unknown_output_lists_actual_graph_choices(transform):
    with pytest.raises(ValueError) as caught:
        transform(_model(), outputs=["loss"])
    _contains(caught.value, "Unknown outputs", "loss", "Available outputs", "result", "exact")


def test_selected_dtype_error_does_not_recommend_casting_away_semantics():
    with pytest.raises(TypeError) as caught:
        jvp(_model(dtype=ir.DataType.INT64), inputs=["x0"])
    _contains(
        caught.value,
        "JVP",
        "x0",
        "INT64",
        "shape=[]",
        "FLOAT or DOUBLE",
        "Select",
        "does not preserve",
    )


def test_grad_shape_error_lists_scalar_choices_and_vector_alternative():
    source = _model(shapes=([1],))
    with pytest.raises(ValueError) as caught:
        grad(source)
    _contains(
        caught.value,
        "result",
        "shape=[1]",
        "rank-zero",
        "scalar FLOAT/DOUBLE",
        "[]",
        "jvp/vjp",
        "intended loss",
    )


@pytest.mark.parametrize(
    "transform,exception",
    [
        (lambda m: vmap(m, [0]), ir.passes.PreconditionError),
        (jvp, ValueError),
        (vjp, ValueError),
        (grad, ValueError),
    ],
)
@pytest.mark.parametrize("version", [17, 29, None])
def test_graph_opset_errors_report_actual_supported_range_and_safe_remedy(
    transform, exception, version
):
    source = _model(opset=version or 18)
    if version is None:
        source.graph.opset_imports.clear()
    with pytest.raises(exception) as caught:
        transform(source)
    _contains(
        caught.value,
        repr(version),
        "18 through 28",
        "diagnostic",
        "validate",
        "relabel",
        "newer",
    )


@pytest.mark.parametrize(
    "transform,exception",
    [
        (jvp, NotImplementedError),
        (vjp, NotImplementedError),
        (lambda m: vmap(m, [0]), ir.passes.PreconditionError),
    ],
)
def test_effective_node_opset_errors_include_supported_range(transform, exception):
    source = _model()
    source.graph[0].version = 29
    with pytest.raises(exception) as caught:
        transform(source)
    _contains(caught.value, "opset 29", "18 through 28", "offending_node", "relabel")
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("transform", [jvp, vjp, lambda m: vmap(m, [0])])
def test_node_schema_override_error_reports_both_versions_and_serialization_limit(transform):
    source = _model("Cast", opset=24, attrs={"to": int(ir.DataType.FLOAT)})
    source.graph[0].version = 18
    with pytest.raises((NotImplementedError, ir.passes.PreconditionError)) as caught:
        transform(source)
    _contains(
        caught.value,
        "ai.onnx::Cast",
        "opset 18",
        "graph opset 24",
        "schema",
        "offending_node",
        "validate",
        "serialization",
    )


@pytest.mark.parametrize("op", ["RandomNormalLike", "Dropout"])
@pytest.mark.parametrize("transform", [jvp, vjp])
def test_random_autodiff_failure_does_not_imply_stochastic_gradients_are_supported(
    op, transform
):
    with pytest.raises(NotImplementedError) as caught:
        transform(_model(op))
    _contains(
        caught.value,
        f"ai.onnx::{op}",
        "opset 18",
        "offending_node",
        "randomness",
        "samples held fixed",
        "stochastic/distributional derivatives are unsupported",
    )


@pytest.mark.parametrize("transform", [jvp, vjp])
def test_subgraph_error_identifies_location_and_supported_boundary(transform):
    body = ir.Graph([], [], nodes=[], opset_imports={"": 18})
    with pytest.raises(NotImplementedError) as caught:
        transform(_model("UnknownControlFlow", attrs={"body": body}))
    _contains(
        caught.value,
        "UnknownControlFlow",
        "opset 18",
        "offending_node",
        "subgraphs",
        "separately",
        "no derivative through branch/loop control",
    )


@pytest.mark.parametrize("transform,mode", [(jvp, "JVP"), (vjp, "VJP"), (grad, "grad (VJP)")])
def test_unsupported_active_operator_is_not_an_invalid_input_or_a_zero_derivative(
    transform, mode
):
    source = _model("Floor")
    before = ir.to_proto(source).SerializeToString()
    with pytest.raises(NotImplementedError) as caught:
        transform(source)
    _contains(
        caught.value,
        mode,
        "ai.onnx::Floor",
        "opset 18",
        "offending_node",
        "active operator",
        "semantics-equivalent",
        "cannot be treated as zero",
    )
    assert isinstance(caught.value.__cause__, NotImplementedError)
    assert ir.to_proto(source).SerializeToString() == before


def test_unsupported_mapped_operator_retains_registry_cause():
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(_model("NonZero", ([2],)), [0])
    _contains(
        caught.value,
        "vmap",
        "ai.onnx::NonZero",
        "opset 18",
        "offending_node",
        "Registered intervals: none",
        "supported operators",
        "semantics",
    )
    assert isinstance(caught.value.__cause__, ir.passes.PreconditionError)
    assert isinstance(caught.value.__cause__.__cause__, NotImplementedError)


@pytest.mark.parametrize("transform,mode", [(jvp, "JVP"), (vjp, "VJP")])
def test_active_cast_error_reports_actual_dtype_and_context(transform, mode):
    source = _model("Cast", attrs={"to": int(ir.DataType.INT64)})
    cast = source.graph[0]
    restored = ir.node("Cast", cast.outputs, {"to": int(ir.DataType.FLOAT)})
    restored.outputs[0].name = "floating"
    source.graph.append(restored)
    source.graph.outputs.clear()
    source.graph.outputs.append(restored.outputs[0])
    with pytest.raises(TypeError) as caught:
        transform(source)
    _contains(
        caught.value,
        mode,
        "ai.onnx::Cast",
        "opset 18",
        "offending_node",
        "result",
        "INT64",
        "Integer/boolean casts",
        "Casting seeds cannot",
    )


def test_shared_control_error_names_only_offending_control():
    source = _model("Reshape", ([2, 2], [1]))
    source.graph.inputs[1].dtype = ir.DataType.INT64
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0, 0])
    _contains(
        caught.value,
        "vmap",
        "ai.onnx::Reshape",
        "opset 18",
        "offending_node",
        "x1",
        "mapped=True",
        "INT64",
        "shape=[N]",
        "independent execution",
    )


def test_runtime_reduction_control_reports_shape_dtype_and_mapping():
    source = _model("ReduceSum", ([2, 2], [None]), attrs={"keepdims": 0})
    source.graph.inputs[1].dtype = ir.DataType.INT64
    source.graph.outputs[0].shape = ir.Shape([None])
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0, None])
    _contains(
        caught.value,
        "ReduceSum",
        "opset 18",
        "offending_node",
        "x1",
        "INT64",
        "mapped=False",
        "known axes length",
        "shape=[N]",
        "intended output",
    )


def test_runtime_cumsum_axis_error_explains_constant_and_noninput_requirements():
    source = _model("CumSum", ([2], ()))
    source.graph.inputs[1].dtype = ir.DataType.INT64
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0, None])
    _contains(
        caught.value,
        "CumSum",
        "opset 18",
        "x1",
        "INT64",
        "rank-zero",
        "non-input initializer/Constant",
        "runtime",
        "unsupported",
    )


def test_training_batchnorm_error_does_not_silently_change_training_semantics():
    source = _model(
        "BatchNormalization", ([1, 1, 2, 2], [1], [1], [1], [1]), attrs={"training_mode": 1}
    )
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0, None, None, None, None])
    _contains(
        caught.value,
        "BatchNormalization",
        "opset 18",
        "training_mode=1",
        "inference mode",
        "only if",
        "independently",
        "change the computation",
    )


def test_rule_axis_error_has_operator_context_before_mutation(monkeypatch):
    # Isolate the rule's guard; engines can reject malformed permutations earlier.
    monkeypatch.setattr(batching, "_infer_shapes", lambda m, **kw: m.clone())
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(_model("Transpose", ([2, 3],), attrs={"perm": [1, 1]}), [0])
    _contains(
        caught.value,
        "vmap",
        "ai.onnx::Transpose",
        "offending_node",
        "opset 18",
        "[1, 1]",
        "[0, 1]",
        "exactly once",
        "example",
    )


def test_rule_rank_error_does_not_dump_initializer_contents(monkeypatch):
    monkeypatch.setattr(batching, "_infer_shapes", lambda m, **kw: m.clone())
    source = _model()
    parameter = ir.val(
        "secret_parameter",
        dtype=ir.DataType.FLOAT,
        shape=[2, 3],
        const_value=ir.tensor(np.full((2, 3), 987654.0, dtype=np.float32)),
    )
    source.graph.register_initializer(parameter)
    source.graph[0].outputs[0].shape = ir.Shape([2])
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0])
    _contains(caught.value, "Identity", "opset 18", "rank", "result", "shape=[2]", "Correct")
    assert "987654" not in str(caught.value)


def test_mapped_initializer_error_describes_metadata_not_tensor_payload():
    source = _model(shapes=([2, 3],))
    value = source.graph.inputs[0]
    value.const_value = ir.tensor(np.full((2, 3), 987654.0, dtype=np.float32))
    source.graph.register_initializer(value)
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(source, [0])
    _contains(caught.value, "x0", "shape=[2, 3]", "initializer defaults", "None", "runtime")
    assert "987654" not in str(caught.value)


@pytest.mark.parametrize(
    "transform,exception",
    [(jvp, ShapeInferenceError), (lambda m: vmap(m, [0, None]), ir.passes.PreconditionError)],
)
def test_inference_boundary_preserves_original_failure_and_adds_action(transform, exception):
    source = _model("MatMul", ([2, 3], [4, 2]))
    before = ir.to_proto(source).SerializeToString()
    with pytest.raises(exception) as caught:
        transform(source)
    error = caught.value if exception is ShapeInferenceError else caught.value.__cause__
    assert isinstance(error, ShapeInferenceError)
    notes = " ".join(error.__notes__)
    assert "diagnostic" in notes and "policy='refine'" in notes
    assert "Correct the reported operator" in notes and "source is unchanged" in notes
    assert "MatMul" in str(error)
    assert ir.to_proto(source).SerializeToString() == before


def test_inference_does_not_wrap_or_swallow_unrelated_library_failure(monkeypatch):
    original = RuntimeError("external tensor data is unavailable; load the external file")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(_inference, "infer_symbolic_shapes", fail)
    with pytest.raises(RuntimeError) as caught:
        jvp(_model())
    assert caught.value is original


@pytest.mark.parametrize("reduction", [None, "mul", "max", "min"])
def test_scatter_composition_guard_never_suggests_switching_semantics(reduction):
    source = _model(
        "ScatterND",
        ([2], [1, 1], [1]),
        attrs=({} if reduction is None else {"reduction": reduction}),
    )
    source.graph.inputs[1].dtype = ir.DataType.INT64
    with pytest.raises(NotImplementedError) as caught:
        jvp(source, inputs=["x0", "x2"])
    _contains(
        caught.value,
        "reduction='add'",
        reduction or "none (default)",
        "separate derivative",
        "changes the function",
    )
    context = " ".join(caught.value.__notes__)
    assert "JVP" in context and "opset 18" in context and "offending_node" in context


@pytest.mark.parametrize(
    "field,bad",
    [
        ("source_inputs", ["missing"]),
        ("source_outputs", "result"),
        ("seeds", [["nested"]]),
        ("derivative_outputs", ["x0_cotangent", "x0_cotangent"]),
    ],
)
def test_serialized_metadata_field_errors_identify_path_value_choices_and_recovery(
    field, bad, tmp_path
):
    source = grad(_model("Exp"))
    info = json.loads(source.graph.metadata_props["onnax.autodiff.interface"])
    info[field] = bad
    source.graph.metadata_props["onnax.autodiff.interface"] = json.dumps(info)
    path = tmp_path / "invalid_interface.onnx"
    ir.save(source, path)
    source = ir.load(path)
    with pytest.raises(NotImplementedError) as caught:
        jvp(source)
    _contains(
        caught.value,
        "Higher-order",
        field,
        repr(bad),
        "unique names",
        "Regenerate from the original",
        "preserve",
        "Do not remove markers",
    )


@pytest.mark.parametrize(
    "bad,fragments",
    [
        ("{", ("Invalid interface metadata", "Expecting", "Regenerate")),
        ("null", ("JSON object", "NoneType", "Regenerate")),
        ('{"version": 9}', ("version=9", "supported version is 1", "Regenerate")),
        ('{"version": 1, "modes": ["jvp"]}', ("modes=['jvp']", "['grad']", "unsupported")),
    ],
)
def test_metadata_parse_and_version_errors_are_specific(bad, fragments):
    source = grad(_model("Exp"))
    source.graph.metadata_props["onnax.autodiff.interface"] = bad
    with pytest.raises(NotImplementedError) as caught:
        jvp(source)
    _contains(caught.value, *fragments)
    if bad == "{":
        assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_reverse_over_reverse_error_reports_requested_mode_and_supported_composition():
    source = grad(_model("Exp"))
    with pytest.raises(NotImplementedError) as caught:
        vjp(source)
    _contains(
        caught.value,
        "Requested VJP",
        "marker='grad'",
        "only one JvpPass",
        "original model",
        "apply jvp once",
        "Do not remove markers",
    )


@pytest.mark.parametrize("orphan", [False, True])
def test_metadata_marker_mismatch_and_orphan_diagnostics_preserve_recovery_contract(orphan):
    source = grad(_model("Exp"))
    if orphan:
        del source.graph.metadata_props["onnax.autodiff"]
    else:
        source.graph.metadata_props["onnax.autodiff"] = "vjp"
    with pytest.raises(NotImplementedError) as caught:
        jvp(source)
    _contains(caught.value, "Higher-order", "marker", "original model", "preserve")


def test_registry_conflict_and_missing_import_diagnostics_are_actionable():
    registry = _rule_registry._RuleRegistry()

    def rule():
        pass

    registry.register("example", "Scale", rule, since_version=3, until_version=5)
    with pytest.raises(ValueError) as caught:
        registry.register("example", "Scale", rule, since_version=5, until_version=6)
    _contains(caught.value, "example", "Scale", "[5, 6]", "[3, 5]", "separate registry")
    node = ir.Node("example", "Scale", [], num_outputs=1, name="custom")
    with pytest.raises(ValueError) as caught:
        registry.lookup(node, {"": 18})
    _contains(
        caught.value,
        "example::Scale",
        "custom",
        "Available imports",
        "18",
        "enclosing graph/function",
        "another domain",
    )
    with pytest.raises(NotImplementedError) as caught:
        registry.lookup(node, {"example": 7})
    _contains(caught.value, "opset 7", "[3, 5]", "registered interval", "relabel")


@pytest.mark.parametrize(
    "argument",
    [
        "axis_size",
        "in_axes",
        "in_axes_entry",
        "in_axes_count",
        "out_axes",
        "out_axes_entry",
        "inputs",
        "nested_inputs",
        "registry_version",
        "graph_version",
    ],
)
def test_malformed_tensor_arguments_never_print_payloads(argument):
    parameter = ir.val(
        "private_parameter",
        dtype=ir.DataType.FLOAT,
        shape=[2],
        const_value=ir.tensor(np.array([987654.0, 123456.0], dtype=np.float32)),
    )
    source = _model()
    actions = {
        "axis_size": lambda: vmap(source, [0], axis_size=parameter),
        "in_axes": lambda: vmap(source, parameter),
        "in_axes_entry": lambda: vmap(source, [parameter]),
        "in_axes_count": lambda: vmap(source, [parameter, None]),
        "out_axes": lambda: vmap(source, [0], out_axes=parameter),
        "out_axes_entry": lambda: vmap(source, [0], out_axes=[parameter]),
        "inputs": lambda: jvp(source, inputs=[parameter]),
        "nested_inputs": lambda: jvp(source, inputs=[[parameter]]),
        "registry_version": lambda: _rule_registry._check_version(parameter),
        "graph_version": lambda: jvp(source),
    }
    if argument == "graph_version":
        source.graph.opset_imports[""] = parameter
    with pytest.raises((TypeError, ValueError, ir.passes.PreconditionError)) as caught:
        actions[argument]()
    text = str(caught.value)
    assert "private_parameter" in text and "shape=[2]" in text
    assert "987654" not in text and "123456" not in text


def test_arbitrary_invalid_argument_repr_is_not_called():
    class InvalidAxis:
        def __repr__(self):
            raise AssertionError("diagnostics must not invoke this representation")

    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(_model(), [InvalidAxis()])
    _contains(caught.value, "<InvalidAxis>", "integer", "in_axes")


@pytest.mark.parametrize("transform", [jvp, lambda m: vmap(m, [0])])
@pytest.mark.parametrize("version,exception", [("18", TypeError), (18.5, ValueError)])
def test_malformed_graph_version_errors_remain_typed_but_become_actionable(
    transform, version, exception
):
    source = _model(opset=version)
    with pytest.raises(exception) as caught:
        transform(source)
    _contains(
        caught.value,
        repr(version),
        "diagnostic",
        "positive integer",
        "18 through 28",
        "Supply",
        "opset",
    )


def test_numpy_argument_and_recursive_sequence_diagnostics_are_bounded():
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(_model(), [np.array([987654, 123456])])
    _contains(caught.value, "<ndarray>", "integer", "in_axes")
    assert "987654" not in str(caught.value)
    axes = []
    axes.append(axes)
    with pytest.raises(ir.passes.PreconditionError) as caught:
        vmap(_model(), axes)
    _contains(caught.value, "<list>", "integer", "in_axes")
    assert len(str(caught.value)) < 500


@pytest.mark.parametrize("transform", [jvp, lambda m: vmap(m, [0])])
@pytest.mark.parametrize("exception", [TypeError, ValueError])
def test_opset_comparison_error_subclass_keeps_original_cause(transform, exception):
    class ComparisonError(exception):
        def __init__(self, left, right):
            super().__init__("invalid comparison")

    original = ComparisonError(18, 28)

    class InvalidOpset:
        def __ge__(self, other):
            raise original

    with pytest.raises(exception) as caught:
        transform(_model(opset=InvalidOpset()))
    assert caught.value.__cause__ is original
    _contains(
        caught.value,
        "<InvalidOpset>",
        "diagnostic",
        "positive integer",
        "18 through 28",
        "validate",
    )
