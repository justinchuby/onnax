# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Operator support sets and graph value accessors."""

from __future__ import annotations

from collections.abc import Sequence

import onnx_ir as ir

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
    if node.op_type == "Where":
        return (1, 2)
    if node.op_type == "ScatterND":
        return (0, 2)
    if node.op_type in _STRUCTURAL | {"Gather", "Split", "ReduceSum", "ReduceMean"}:
        return (0,)
    return range(len(node.inputs))


def _input(node: ir.Node, index: int = 0) -> ir.Value:
    if index >= len(node.inputs) or node.inputs[index] is None:
        raise ValueError(f"{node.op_type} requires input {index}.")
    value = node.inputs[index]
    assert value is not None
    return value


def _int(node: ir.Node, name: str, default: int) -> int:
    attr = node.attributes.get(name)
    return default if attr is None else attr.as_int()


def _float(node: ir.Node, name: str, default: float) -> float:
    attr = node.attributes.get(name)
    return default if attr is None else attr.as_float()


def _rank(value: ir.Value) -> int:
    if value.shape is None:
        raise ValueError(
            f"Autodiff requires a known rank for {value.name!r}; "
            "provide shape metadata or run shape inference first."
        )
    return len(value.shape)
