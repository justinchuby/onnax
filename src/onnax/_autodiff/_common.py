# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Operator support sets and graph value accessors."""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir

from onnax._diagnostics import _value_context

_FLOATS = frozenset((ir.DataType.FLOAT, ir.DataType.DOUBLE))

_MARKER = "onnax.autodiff"

_INTERFACE = "onnax.autodiff.interface"

_BINARY = frozenset(("Add", "Sub", "Mul", "Div", "Pow"))


_STRUCTURAL = frozenset(("Reshape", "Flatten", "Squeeze", "Unsqueeze", "Expand"))


_NO_DATA = frozenset(
    (
        "Constant",
        "ConstantOfShape",
        "Shape",
        "Size",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
        "Not",
        "IsInf",
        "IsNaN",
    )
)


def _data_indices(node: ir.Node) -> Sequence[int]:
    if node.domain:
        return range(len(node.inputs))
    if node.op_type in _NO_DATA:
        return ()
    if node.op_type == "Attention":
        return range(min(4, len(node.inputs)))
    if node.op_type == "RotaryEmbedding":
        return range(min(3, len(node.inputs)))
    if node.op_type == "Where":
        return (1, 2)
    if node.op_type == "Pad":
        return (0, 2) if len(node.inputs) > 2 and node.inputs[2] is not None else (0,)
    if node.op_type == "ScatterND":
        return (0, 2)
    if node.op_type in _STRUCTURAL | {
        "Gather",
        "Slice",
        "Split",
        "ReduceSum",
        "ReduceMean",
        "ReduceL1",
        "ReduceLogSum",
        "ReduceSumSquare",
    }:
        return (0,)
    return range(len(node.inputs))


def _input(node: ir.Node, index: int = 0) -> ir.Value:
    if index >= len(node.inputs) or node.inputs[index] is None:
        raise ValueError(
            f"{node.op_type} requires input {index}; got {len(node.inputs)} slots, "
            f"present indices={[i for i, v in enumerate(node.inputs) if v is not None]!r} "
            f"(node {node.name!r}). Supply the required operand in its ONNX schema slot; "
            "retain empty slots only for optional inputs."
        )
    value = node.inputs[index]
    assert value is not None
    return value


def _int(node: ir.Node, name: str, default: int) -> int:
    attr = node.attributes.get(name)
    return default if attr is None else attr.as_int()


def _float(node: ir.Node, name: str, default: float) -> float:
    attr = node.attributes.get(name)
    value = default if attr is None else attr.as_float()
    # AttributeProto.FLOAT is binary32, independently of the tensor dtype.
    return ir.tensor(value, dtype=ir.DataType.FLOAT).numpy().item()


def _rank(value: ir.Value) -> int:
    if value.shape is None:
        raise ValueError(
            f"Autodiff requires a known rank for {_value_context(value)}; "
            "annotate this value's source shape with one entry per dimension "
            "(symbolic dimensions are allowed). Inference could not determine its rank."
        )
    return len(value.shape)
