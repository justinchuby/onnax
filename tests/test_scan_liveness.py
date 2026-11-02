# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Hand-calculated Scan carry, capture and output-buffer schedules."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from examples.activation_liveness import analyze


def scan_graph(
    length, *, axis=0, reverse=0, capture=False, initializer=False, passthrough=False
):
    carry = ir.val("carry", dtype=ir.DataType.FLOAT, shape=[4])
    item = ir.val("item", dtype=ir.DataType.FLOAT, shape=[4])
    inputs = [
        ir.val("initial", dtype=ir.DataType.FLOAT, shape=[4]),
        ir.val(
            "sequence",
            dtype=ir.DataType.FLOAT,
            shape=[length, 4] if axis == 0 else [4, length],
        ),
    ]
    outer = ir.tape.Tape()
    bias = None
    if capture:
        source = ir.val("source", dtype=ir.DataType.FLOAT, shape=[4])
        inputs.append(source)
        bias = outer.op("Neg", [source])
        bias.name, bias.dtype, bias.shape = "captured", ir.DataType.FLOAT, ir.Shape([4])
    inner = ir.tape.Tape()
    if initializer:
        bias = inner.initializer(ir.tensor(np.ones(4, np.float32), name="bias"))
    added = inner.op("Add", [carry, item])
    state = inner.op("Identity", [carry]) if passthrough else inner.op("Neg", [added])
    emitted = inner.op("Identity", [added]) if bias is None else inner.op("Add", [added, bias])
    for name, value in (("added", added), ("next_state", state), ("emitted", emitted)):
        value.name, value.dtype, value.shape = name, ir.DataType.FLOAT, ir.Shape([4])
    body = ir.Graph(
        [carry, item],
        [state, emitted],
        nodes=inner.nodes,
        initializers=inner.initializers,
        opset_imports={"": 24},
        name="scan_body",
    )
    last, stacked = outer.op_multi_out(
        "Scan",
        inputs[:2],
        {
            "body": body,
            "num_scan_inputs": 1,
            "scan_input_axes": [axis],
            "scan_output_axes": [axis],
            "scan_input_directions": [reverse],
            "scan_output_directions": [1 - reverse],
        },
        num_outputs=2,
    )
    last.name, last.dtype, last.shape = "last", ir.DataType.FLOAT, ir.Shape([4])
    stacked.name, stacked.dtype, stacked.shape = (
        "stacked",
        ir.DataType.FLOAT,
        ir.Shape(inputs[1].shape),
    )
    return ir.Model(
        ir.Graph(
            inputs, [last, stacked], nodes=outer.nodes, opset_imports={"": 24}, name="scan"
        ),
        ir_version=12,
    )


@pytest.mark.parametrize("length,total", [(0, 32), (1, 96), (3, 176)])
def test_scan_hand_calculated_peak_and_source_preservation(length, total):
    model = scan_graph(length)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    before = ir.to_proto(model).SerializeToString()
    result = analyze(model, scan_bodies=True)
    assert ir.to_proto(model).SerializeToString() == before
    assert result["peak_total_tensor_bytes"] == total
    detail = result["scans"][0]
    assert detail["steps"] == length
    assert detail["carry_bytes"] == 16
    assert detail["retained_sequence_output_bytes"] == length * 16
    assert detail["borrowed_slice_bytes"] == 16
    assert detail["previous_iteration_carry_bytes"] == (16 if length > 1 else 0)
    assert detail["body_temporary_peak_bytes"] == 48
    assert (
        result["peak_activation_bytes"]
        == sum(group["bytes"] for group in result["peak_groups"])
        + result["peak_scan_workspace_bytes"]
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("reverse", [0, 1])
def test_nonzero_axes_and_directions_have_identical_storage(axis, reverse):
    model = scan_graph(3, axis=axis, reverse=reverse)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    assert analyze(model, scan_bodies=True)["peak_total_tensor_bytes"] == 176


def test_body_view_alias_and_previous_carry_accounting():
    result = analyze(scan_graph(3), scan_bodies=True, alias_views=True)
    assert result["scans"][0]["body_temporary_peak_bytes"] == 32
    assert result["peak_total_tensor_bytes"] == 160


def test_passthrough_carry_keeps_borrowed_storage():
    result = analyze(scan_graph(3, passthrough=True), scan_bodies=True, alias_views=True)
    assert result["scans"][0]["previous_iteration_carry_bytes"] == 0
    assert result["peak_total_tensor_bytes"] == 128


def test_cross_state_aliases_retain_older_owned_buffers():
    a, b, item = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=[4]) for name in ("a", "b", "item")
    ]
    inner = ir.tape.Tape()
    next_a = inner.op("Identity", [b])
    next_b = inner.op("Neg", [b])
    for name, value in (("next_a", next_a), ("next_b", next_b)):
        value.name, value.dtype, value.shape = name, ir.DataType.FLOAT, ir.Shape([4])
    body = ir.Graph([a, b, item], [next_a, next_b], nodes=inner.nodes, opset_imports={"": 24})
    initial = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=[4]) for name in ("initial_a", "initial_b")
    ]
    sequence = ir.val("sequence", dtype=ir.DataType.FLOAT, shape=[3, 4])
    tape = ir.tape.Tape()
    results = tape.op_multi_out(
        "Scan", [*initial, sequence], {"body": body, "num_scan_inputs": 1}, num_outputs=2
    )
    for value in results:
        value.dtype, value.shape = ir.DataType.FLOAT, ir.Shape([4])
    model = ir.Model(
        ir.Graph([*initial, sequence], results, nodes=tape.nodes, opset_imports={"": 24}),
        ir_version=12,
    )
    result = analyze(model, scan_bodies=True, alias_views=True)
    # Before iteration three, A and B refer to different earlier Neg results.
    assert result["scans"][0]["previous_iteration_carry_bytes"] == 32
    assert result["peak_total_tensor_bytes"] == 128


def test_capture_is_live_through_its_nested_use():
    model = scan_graph(3, capture=True)
    before = ir.to_proto(model).SerializeToString()
    result = analyze(model, scan_bodies=True)
    assert result["peak_total_tensor_bytes"] == 208
    capture = next(row for row in result["values"] if row["name"] == "captured")
    assert capture["birth"] == 0 and capture["last_use"] == 1
    assert result["scans"][0]["captures"] == ["captured"]
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("length,total", [(0, 32), (3, 192)])
def test_body_initializers_allocated_once_when_loop_runs(length, total):
    result = analyze(scan_graph(length, initializer=True), scan_bodies=True)
    assert result["peak_total_tensor_bytes"] == total
    assert result["scans"][0]["body_initializer_bytes"] == 16


def test_nested_scan_combines_per_iteration_and_retained_buffers():
    inner = scan_graph(3)
    body = inner.graph
    initial = ir.val("outer_initial", dtype=ir.DataType.FLOAT, shape=[4])
    sequence = ir.val("outer_sequence", dtype=ir.DataType.FLOAT, shape=[2, 3, 4])
    tape = ir.tape.Tape()
    last, output = tape.op_multi_out(
        "Scan", [initial, sequence], {"body": body, "num_scan_inputs": 1}, num_outputs=2
    )
    last.dtype, last.shape = ir.DataType.FLOAT, ir.Shape([4])
    output.dtype, output.shape = ir.DataType.FLOAT, ir.Shape([2, 3, 4])
    model = ir.Model(
        ir.Graph(
            [initial, sequence], [last, output], nodes=tape.nodes, opset_imports={"": 24}
        ),
        ir_version=12,
    )
    result = analyze(model, scan_bodies=True)
    assert result["peak_total_tensor_bytes"] == 336
    assert len(result["scans"][0]["body"]["scans"]) == 1


def test_scan_with_only_final_carry_allocates_no_sequence_output():
    carry = ir.val("carry", dtype=ir.DataType.FLOAT, shape=[4])
    item = ir.val("item", dtype=ir.DataType.FLOAT, shape=[4])
    inner = ir.tape.Tape()
    state = inner.op("Add", [carry, item])
    state.name, state.dtype, state.shape = "next", ir.DataType.FLOAT, ir.Shape([4])
    body = ir.Graph(
        [carry, item], [state], nodes=inner.nodes, opset_imports={"": 24}, name="step"
    )
    initial = ir.val("initial", dtype=ir.DataType.FLOAT, shape=[4])
    sequence = ir.val("sequence", dtype=ir.DataType.FLOAT, shape=[3, 4])
    tape = ir.tape.Tape()
    output = tape.op("Scan", [initial, sequence], {"body": body, "num_scan_inputs": 1})
    output.name, output.dtype, output.shape = "last", ir.DataType.FLOAT, ir.Shape([4])
    model = ir.Model(
        ir.Graph(
            [initial, sequence],
            [output],
            nodes=tape.nodes,
            opset_imports={"": 24},
            name="carry_only",
        ),
        ir_version=12,
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    result = analyze(model, scan_bodies=True)
    assert result["peak_total_tensor_bytes"] == 96
    assert result["scans"][0]["retained_sequence_output_bytes"] == 0


def test_invalid_axes_and_carry_shapes_are_rejected():
    model = scan_graph(3)
    node = next(iter(model.graph))
    node.attributes["scan_input_axes"] = ir.AttrInt64s("scan_input_axes", [2])
    with pytest.raises(ValueError, match="axis"):
        analyze(model, scan_bodies=True)
    model = scan_graph(3)
    body = next(iter(model.graph)).attributes.get_graph("body")
    body.inputs[0].shape = ir.Shape([5])
    with pytest.raises(ValueError, match="carry"):
        analyze(model, scan_bodies=True)


def test_default_retains_flat_graph_contract():
    with pytest.raises(ValueError, match="control flow"):
        analyze(scan_graph(3))


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_half_precision_storage_uses_the_graph_dtype(dtype):
    model = scan_graph(3)
    for graph in model.graphs():
        for value in ir.convenience.create_value_mapping(graph).values():
            value.dtype = dtype
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    result = analyze(model, scan_bodies=True)
    assert result["peak_total_tensor_bytes"] == 88
    assert result["scans"][0]["carry_bytes"] == 8


def test_legacy_batched_scan_requires_a_different_memory_model():
    model = scan_graph(3)
    model.graph.opset_imports[""] = 8
    with pytest.raises(ValueError, match="opset 9"):
        analyze(model, scan_bodies=True)
