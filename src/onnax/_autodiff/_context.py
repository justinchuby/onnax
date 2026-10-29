# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Derivative graph construction and runtime broadcast reduction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnx_ir as ir

from ._common import _FLOATS


class _Context:
    def __init__(self, graph: ir.Graph) -> None:
        self.graph = graph
        self.tape = ir.tape.Tape(graph)
        self.names = set(ir.convenience.create_value_mapping(graph, include_subgraphs=False))

    def name(self, stem: str) -> str:
        name = stem
        suffix = 1
        while name in self.names:
            name = f"{stem}_{suffix}"
            suffix += 1
        self.names.add(name)
        return name

    def op(
        self,
        op: str,
        *inputs: ir.Value | None,
        **attrs: ir.Attr | int | float | str | Sequence[int] | ir.TensorProtocol,
    ) -> ir.Value:
        return self.tape.op(op, inputs, attrs)

    def constant(
        self, value: float | int | Sequence[int], dtype: ir.DataType = ir.DataType.INT64
    ) -> ir.Value:
        return self.op("Constant", value=ir.tensor(np.asarray(value, dtype=dtype.numpy())))

    def scalar(self, reference: ir.Value, value: float) -> ir.Value:
        assert reference.dtype in _FLOATS
        return self.constant(value, reference.dtype)

    def like(self, reference: ir.Value, fill: float = 0.0) -> ir.Value:
        assert reference.dtype in _FLOATS
        return self.op(
            "ConstantOfShape",
            self.op("Shape", reference),
            value=ir.tensor([fill], dtype=reference.dtype),
        )

    def reshape(self, value: ir.Value, shape: ir.Value) -> ir.Value:
        return self.op("Reshape", value, shape, allowzero=1)

    def sum(self, values: Sequence[ir.Value]) -> ir.Value:
        result = values[0]
        for value in values[1:]:
            result = self.op("Add", result, value)
        return result

    def seed(self, value: ir.Value, suffix: str) -> ir.Value:
        assert value.dtype is not None
        result = ir.Value(
            name=self.name(f"{value.name}_{suffix}"),
            type=ir.TensorType(value.dtype),
            shape=ir.Shape(value.shape) if value.shape is not None else None,
        )
        self.graph.inputs.append(result)
        return result

    def output(self, derivative: ir.Value, primal: ir.Value, suffix: str) -> None:
        assert primal.dtype is not None
        output = self.op("Identity", derivative)
        output.name = self.name(f"{primal.name}_{suffix}")
        output.type = ir.TensorType(primal.dtype)
        output.shape = ir.Shape(primal.shape) if primal.shape is not None else None
        self.graph.outputs.append(output)

    def unbroadcast(self, gradient: ir.Value, primal: ir.Value) -> ir.Value:
        """Sum runtime singleton and leading broadcast dimensions, including zero extents."""
        shape = self.op("Shape", primal)
        grad_shape = self.op("Shape", gradient)
        leading = self.op("Sub", self.op("Size", grad_shape), self.op("Size", shape))
        ones = self.op(
            "ConstantOfShape",
            self.op("Unsqueeze", leading, self.constant([0])),
            value=ir.tensor([1], dtype=ir.DataType.INT64),
        )
        padded = self.op("Concat", ones, shape, axis=0)
        axes = self.op(
            "Squeeze",
            self.op("NonZero", self.op("Equal", padded, self.constant(1))),
            self.constant([0]),
        )
        # A dummy singleton axis avoids runtime bugs in empty-axis reductions of empty tensors.
        axes = self.op("Concat", axes, self.constant([-1]), axis=0)
        expanded = self.op("Unsqueeze", gradient, self.constant([-1]))
        reduced = self.op("ReduceSum", expanded, axes, keepdims=1)
        return self.reshape(reduced, shape)

    def transpose(self, value: ir.Value, rank: int) -> ir.Value:
        perm = list(range(rank))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.op("Transpose", value, perm=perm)
