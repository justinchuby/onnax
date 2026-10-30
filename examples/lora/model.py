# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Example-specific adaptation and stable runtime interface for a loaded ONNX base."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import onnx
import onnx_ir as ir

import onnax

PARAMETERS = ("lora.A", "lora.B")


def base_model(w, c):
    if w.ndim != 2 or c.ndim != 2 or w.shape[1] != c.shape[0] or min(*w.shape, *c.shape) < 1:
        raise ValueError("Expected compatible nonempty rank-two base weights")
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["batch", w.shape[0]])
    tape = ir.tape.Tape()
    weight = tape.initializer(ir.tensor(w, name="base.W"))
    head = tape.initializer(ir.tensor(c, name="base.C"))
    hidden = tape.op("Tanh", [tape.op("MatMul", [x, weight])])
    prediction = tape.op("MatMul", [hidden, head])
    prediction.name = "prediction"
    prediction.type = ir.TensorType(ir.DataType.FLOAT)
    prediction.shape = ir.Shape(["batch", c.shape[1]])
    return ir.Model(
        ir.Graph(
            [x],
            [prediction],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="frozen_base",
        ),
        ir_version=10,
    )


def adapt(base, *, rank=4):
    """Add a rank-controlled LoRA branch to the XW -> Tanh -> C graph."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    result = base.clone()
    graph = result.graph
    names = ir.convenience.create_value_mapping(graph, include_subgraphs=False)
    reserved = {*PARAMETERS, "target", "loss", "lora.scale"}
    if reserved.intersection(names):
        raise ValueError(
            f"LoRA reserved name collision: {sorted(reserved.intersection(names))}"
        )
    if set(graph.initializers) != {"base.W", "base.C"} or len(graph.inputs) != 1:
        raise ValueError("Expected the example's frozen base.W/base.C model.")
    nodes = list(graph)
    if [n.op_type for n in nodes] != ["MatMul", "Tanh", "MatMul"]:
        raise ValueError("Expected the example's MatMul/Tanh/MatMul base architecture.")
    shapes = {}
    for name in ("base.W", "base.C"):
        value = graph.initializers[name]
        if (
            value.dtype != ir.DataType.FLOAT
            or value.shape is None
            or len(value.shape) != 2
            or any(not isinstance(d, int) or d < 1 for d in value.shape)
        ):
            raise ValueError(f"Expected a static nonempty FLOAT matrix for {name}.")
        shapes[name] = list(value.shape)
    width, hidden = shapes["base.W"]
    if shapes["base.C"][0] != hidden:
        raise ValueError("Frozen base.W/base.C inner dimensions must match")
    x = graph.inputs[0]
    if (
        x.name != "features"
        or x.dtype != ir.DataType.FLOAT
        or x.shape is None
        or len(x.shape) != 2
        or x.shape[1] != width
        or nodes[0].inputs[0] is not x
        or nodes[0].inputs[1] is not graph.initializers["base.W"]
        or nodes[1].inputs[0] is not nodes[0].outputs[0]
        or nodes[2].inputs[0] is not nodes[1].outputs[0]
        or nodes[2].inputs[1] is not graph.initializers["base.C"]
        or list(graph.outputs) != [nodes[2].outputs[0]]
        or graph.outputs[0].name != "prediction"
    ):
        raise ValueError(
            "Expected features -> MatMul(base.W) -> Tanh -> MatMul(base.C) -> prediction."
        )
    a = ir.val(PARAMETERS[0], dtype=ir.DataType.FLOAT, shape=[width, rank])
    b = ir.val(PARAMETERS[1], dtype=ir.DataType.FLOAT, shape=[rank, hidden])
    graph.inputs.extend([a, b])
    tape = ir.tape.Tape()
    scale = tape.initializer(ir.tensor(np.array(0.25, np.float32), name="lora.scale"))
    delta = tape.op("Mul", [tape.op("MatMul", [tape.op("MatMul", [x, a]), b]), scale])
    combined = tape.op("Add", [nodes[0].outputs[0], delta])
    nodes[1].replace_input_with(0, combined)
    graph.insert_before(nodes[1], tape.nodes)
    for value in tape.initializers:
        graph.register_initializer(value)
    return result


def with_loss(forward):
    result = forward.clone()
    prediction = result.graph.outputs[0]
    if prediction.shape is None or len(prediction.shape) != 2:
        raise ValueError("Expected a rank-two prediction for mean squared loss")
    target = ir.val("target", dtype=ir.DataType.FLOAT, shape=ir.Shape(prediction.shape))
    result.graph.inputs.append(target)
    tape = ir.tape.Tape(result.graph)
    residual = tape.op("Sub", [result.graph.outputs[0], target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "loss"
    loss.type = ir.TensorType(ir.DataType.FLOAT)
    loss.shape = ir.Shape([])
    result.graph.outputs.append(loss)
    return result


def digest(value):
    return hashlib.sha256(value.const_value.numpy().tobytes()).hexdigest()


def export(base_path, output_dir):
    base = ir.load(base_path)
    before = ir.to_proto(base).SerializeToString()
    forward = adapt(base)
    loss = with_loss(forward)
    gradient = onnax.grad(loss, inputs=list(PARAMETERS), output="loss")
    mapping = dict(
        zip(
            PARAMETERS,
            [v.name for v in gradient.graph.outputs[len(loss.graph.outputs) :]],
            strict=True,
        )
    )
    input_shapes = {value.name: list(value.shape) for value in forward.graph.inputs}
    interface = {
        "data_shapes": {
            "features": [None, input_shapes["features"][1]],
            "target": [None, loss.graph.inputs[-1].shape[1]],
        },
        "version": 1,
        "loss": "loss",
        "loss_reduction": "mean over batch and outputs",
        "parameters": [
            {"input": name, "gradient": mapping[name], "shape": shape, "dtype": "float32"}
            for name, shape in ((name, input_shapes[name]) for name in PARAMETERS)
        ],
        "frozen": {
            name: digest(base.graph.initializers[name]) for name in ("base.W", "base.C")
        },
    }
    for name, model in (("forward", forward), ("loss", loss), ("gradient", gradient)):
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        ir.save(model, output_dir / f"{name}.onnx")
    (output_dir / "interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    assert ir.to_proto(base).SerializeToString() == before
    return forward, loss, gradient, interface


def validate_interface(model, interface):
    if interface.get("version") != 1:
        raise ValueError("Unsupported LoRA interface version.")
    inputs = {v.name: v for v in model.graph.inputs}
    outputs = {v.name: v for v in model.graph.outputs}
    if (
        interface.get("loss") != "loss"
        or interface.get("loss_reduction") != "mean over batch and outputs"
    ):
        raise ValueError("Loss interface mismatch.")
    if "loss" not in outputs or list(outputs["loss"].shape or []) != []:
        raise ValueError("Expected scalar loss output.")
    if set(interface.get("frozen", {})) != {"base.W", "base.C"}:
        raise ValueError("Expected exactly frozen base.W/base.C weights.")
    rows = interface.get("parameters", [])
    if any(not {"input", "gradient", "shape", "dtype"}.issubset(row) for row in rows):
        raise ValueError("Incomplete parameter interface.")
    if [row["input"] for row in rows] != list(PARAMETERS):
        raise ValueError("Expected exactly ordered lora.A/lora.B parameter mappings.")
    if len({row["gradient"] for row in rows}) != len(rows):
        raise ValueError("Gradient names must be unique.")
    for row in rows:
        for key, values in (("input", inputs), ("gradient", outputs)):
            value = values.get(row[key])
            if (
                value is None
                or list(value.shape or []) != row["shape"]
                or value.dtype != ir.DataType.FLOAT
                or row["dtype"] != "float32"
            ):
                raise ValueError(f"Interface mismatch for {row[key]}.")
        if row["input"] in model.graph.initializers:
            raise ValueError("Trainable parameters must not have initializer defaults.")
    for name, expected in interface["frozen"].items():
        if (
            name not in model.graph.initializers
            or digest(model.graph.initializers[name]) != expected
        ):
            raise ValueError(f"Frozen weight mismatch: {name}.")
