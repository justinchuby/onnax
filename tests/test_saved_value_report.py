# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Saved-value accounting follows derivative uses, with explicit unknown sizes."""

import hashlib
import json
import sys

import numpy as np
import onnx_ir as ir
import pytest

import onnax
from examples.huggingface.saved_values import main, report
from onnax import _rules


def polynomial(*, return_saved=False, batch="batch"):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[batch, 4])
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=[4])
    unused = ir.val("unused", dtype=ir.DataType.FLOAT, shape=[batch, 4])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(np.ones((4, 2), np.float32), name="w"))
    tape.initializer(ir.tensor(np.ones((7,), np.float32), name="unused_weight"))
    u = tape.op("Mul", [x, a])
    square = tape.op("Mul", [u, u])
    y = tape.op("MatMul", [square, w])
    for name, value, shape in (
        ("u", u, [batch, 4]),
        ("square", square, [batch, 4]),
        ("y", y, [batch, 2]),
    ):
        value.name, value.dtype, value.shape = name, ir.DataType.FLOAT, ir.Shape(shape)
    return ir.Model(
        ir.Graph(
            [x, a, unused],
            [u, y] if return_saved else [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 24},
            name="polynomial",
        ),
        ir_version=12,
    )


@pytest.mark.parametrize("return_saved", [False, True])
def test_actual_reads_unique_storage_and_input_preservation(return_saved):
    model = polynomial(return_saved=return_saved, batch=3)
    before = ir.to_proto(model).SerializeToString()
    result = report(
        model, inputs=["a"], outputs=["y"], parameters=["a"], dimensions={"batch": 3}
    )
    assert ir.to_proto(model).SerializeToString() == before
    values = {row["name"]: row for row in result["values"]}
    assert set(values) == {"u", "x", "w", "y_cotangent"}
    assert values["u"]["category"] == "saved_activation"
    assert values["u"]["producer"]["op_type"] == "Mul"
    assert values["u"]["bytes"] == 48
    assert len(values["u"]["backward_consumers"]) == 2
    assert values["x"]["category"] == "data_input"
    assert values["w"]["category"] == "frozen_initializer"
    assert values["y_cotangent"]["category"] == "cotangent"
    assert result["categories"]["saved_activation"] == {
        "count": 1,
        "known_bytes": 48,
        "unknown_values": [],
    }
    assert result["extra_forward_outputs"]["known_bytes"] == (0 if return_saved else 48)
    live = result["backward_liveness"]
    assert live["status"] == "complete"
    assert live["persistent_external_bytes"] == 48 + 48 + 32 + 24
    # Three 48-byte pullback temporaries coexist before last-use release.
    assert live["peak_activation_bytes"] == 144
    assert live["peak_total_tensor_bytes"] == 296
    assert sum(live["peak_by_category"].values()) == 296
    assert not {"unused", "a", "unused_weight"} & {row["name"] for row in live["values"]}


def test_unknown_dimensions_are_reported_without_a_false_total():
    result = report(polynomial(), inputs=["a"], outputs=["y"], parameters=["a"])
    saved = result["categories"]["saved_activation"]
    assert saved["known_bytes"] == 0 and saved["unknown_values"] == ["u"]
    assert result["backward_liveness"]["status"] == "unresolved"
    assert "static" in result["backward_liveness"]["reason"]
    assert result["extra_forward_outputs"]["unknown_values"] == ["u"]


def test_initializer_backed_parameter_is_counted_once():
    model = polynomial()
    parameter = model.graph.inputs[1]
    parameter.const_value = ir.tensor(np.ones(4, np.float32), name="a")
    model.graph.initializers["a"] = parameter
    result = report(model, inputs=["x"], outputs=["y"], dimensions={"batch": 3})
    assert len([row for row in result["values"] if row["name"] == "a"]) == 1
    assert (
        next(row for row in result["values"] if row["name"] == "a")["category"] == "data_input"
    )


def test_optimized_causal_attention_has_concrete_mask_storage():
    values = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=[1, 1, 3, 2]) for name in ("q", "k", "v")
    ]
    tape = ir.tape.Tape()
    y = tape.op("Attention", values, {"is_causal": 1})
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape([1, 1, 3, 2])
    model = ir.Model(
        ir.Graph(values, [y], nodes=tape.nodes, opset_imports={"": 24}), ir_version=12
    )
    result = report(model, inputs=["q"], outputs=["y"], optimize=True)
    assert result["optimization"] == "training_pipeline"
    assert result["backward_liveness"]["status"] == "complete"
    assert all(row["bytes"] is not None for row in result["values"])


def test_unused_functions_do_not_block_flat_liveness():
    model = polynomial(batch=3)
    value = ir.val("arg", dtype=ir.DataType.FLOAT, shape=[1])
    tape = ir.tape.Tape()
    result = tape.op("Neg", [value])
    function = ir.Function(
        "pkg.onnax",
        "Unused",
        graph=ir.Graph([value], [result], nodes=tape.nodes, opset_imports={"": 24}),
        attributes=[],
    )
    model.functions[function.identifier()] = function
    result = report(model, inputs=["a"], outputs=["y"], dimensions={"batch": 3})
    assert result["backward_liveness"]["status"] == "complete"
    assert function.identifier() in model.functions


def test_generated_scan_reports_its_control_flow_analysis_requirement():
    shapes = {
        "q": [1, 3, 2],
        "k": [1, 3, 2],
        "v": [1, 3, 2],
        "state": [1, 1, 2, 2],
        "decay": [1, 3, 1],
        "beta": [1, 3, 1],
    }
    inputs = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape) for name, shape in shapes.items()
    ]
    tape = ir.tape.Tape()
    y, state = tape.op_multi_out(
        "LinearAttention",
        inputs,
        {"q_num_heads": 1, "kv_num_heads": 1, "scale": 0.7, "update_rule": "gated_delta"},
        domain="com.microsoft",
        num_outputs=2,
    )
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape([1, 3, 2])
    state.name, state.dtype, state.shape = (
        "last_state",
        ir.DataType.FLOAT,
        ir.Shape([1, 1, 2, 2]),
    )
    model = ir.Model(
        ir.Graph(
            inputs, [y, state], nodes=tape.nodes, opset_imports={"": 24, "com.microsoft": 1}
        ),
        ir_version=12,
    )
    result = report(model, inputs=["q"], outputs=["y"], optimize=True)
    assert result["backward_liveness"]["status"] == "unresolved"
    assert "control flow" in result["backward_liveness"]["reason"]


def test_specialization_preserves_the_exported_derivative_selection(monkeypatch):
    monkeypatch.setattr(_rules, "_EXTENSIONS", _rules._snapshot())
    observed = []

    @onnax.register_vjp("memory.tests", "Identity", since_version=1, until_version=1)
    def identity(ctx, node, cotangents, indices):
        observed.append(str(node.inputs[0].shape[0]))
        return dict.fromkeys(indices, cotangents[0])

    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 4])
    tape = ir.tape.Tape()
    y = tape.op("Identity", [x], domain="memory.tests")
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape(["batch", 4])
    model = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 24, "memory.tests": 1}),
        ir_version=12,
    )
    result = report(model, inputs=["x"], outputs=["y"], dimensions={"batch": 3})
    assert observed == ["batch"]
    assert result["backward_liveness"]["status"] == "complete"


def test_bundle_cli_uses_declared_block_and_parameter_mapping(tmp_path, monkeypatch):
    path = tmp_path / "block_0_forward.onnx"
    ir.save(polynomial(batch=3), path)
    interface = {
        "version": 1,
        "kind": "checkpointed_vjp",
        "identity": "fixture",
        "specification": {"parameters": ["a"]},
        "blocks": [
            {
                "name": "decoder",
                "active_inputs": ["a"],
                "active_outputs": ["y"],
                "backward": True,
            }
        ],
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (tmp_path / "interface.json").write_text(json.dumps(interface))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saved_values",
            "--bundle",
            str(tmp_path),
            "--block",
            "decoder",
            "--dimension",
            "batch=3",
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads(output.read_text())
    assert result["provenance"]["block"] == "decoder"
    assert result["categories"]["saved_activation"]["known_bytes"] == 48
    assert result["backward_liveness"]["peak_total_tensor_bytes"] == 296


@pytest.mark.parametrize(
    "dimensions", [{"batch": -1}, {"batch": True}, {"": 2}, {"batch": 2.5}]
)
def test_reject_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="Dimensions"):
        report(polynomial(), inputs=["a"], outputs=["y"], dimensions=dimensions)


@pytest.mark.parametrize("parameters", [["missing"], ["unused"]])
def test_parameters_must_be_selected_explicit_inputs(parameters):
    with pytest.raises(ValueError, match="parameter"):
        report(polynomial(), inputs=["a"], outputs=["y"], parameters=parameters)
