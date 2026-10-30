# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Forward and reverse rules for reductions operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _FLOATS, _input, _int
from onnax._autodiff._context import _Context
from onnax._rules import _builtin

OPS = frozenset(
    ("LogSoftmax", "ReduceL1", "ReduceMean", "ReduceSum", "ReduceSumSquare", "Softmax")
)


def reduction_vjp(ctx: _Context, node: ir.Node, g: ir.Value) -> ir.Value:
    x = _input(node)
    shape = ctx.op("Shape", x)
    axes = node.inputs[1] if len(node.inputs) > 1 else None
    # Kept dimensions already broadcast. Omitted axes either reduce everything
    # to a scalar or preserve the input shape (noop), both directly expandable.
    axes_count = (
        axes.shape[0]
        if axes is not None and axes.shape is not None and len(axes.shape) == 1
        else None
    )
    # Empty axes reduce all dimensions or none; either result broadcasts directly.
    if not _int(node, "keepdims", 1) and axes is not None and axes_count != 0:
        if not _int(node, "noop_with_empty_axes", 0) and not isinstance(axes_count, int):
            rank = ctx.op("Size", shape)
            zero = ctx.constant(0)
            stop = ctx.op("Where", ctx.op("Equal", ctx.op("Size", axes), zero), rank, zero)
            axes = ctx.op("Concat", axes, ctx.op("Range", zero, stop, ctx.constant(1)), axis=0)
        ones = ctx.op(
            "ConstantOfShape",
            ctx.op("Shape", axes),
            value=ir.tensor([1], dtype=ir.DataType.INT64),
        )
        keep_shape = ctx.op("ScatterElements", shape, axes, ones, axis=0)
        g = ctx.reshape(g, keep_shape)
    result = ctx.op("Expand", g, shape)
    if node.op_type == "ReduceMean":
        output_dtype = node.outputs[0].dtype
        assert x.dtype in _FLOATS and output_dtype in _FLOATS
        count = ctx.op(
            "Div",
            ctx.op("Cast", ctx.op("Size", x), to=x.dtype),
            ctx.op("Cast", ctx.op("Size", node.outputs[0]), to=output_dtype),
        )
        result = ctx.op("Div", result, count)
    return result


@_builtin("jvp", OPS)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    op = node.op_type
    if op in ("ReduceMean", "ReduceSum"):
        return [ctx.tape.op(op, [d[0], *node.inputs[1:]], attributes=dict(node.attributes))]
    elif op in ("ReduceL1", "ReduceSumSquare"):
        x = _input(node)
        slope = ctx.op("Sign", x) if op == "ReduceL1" else ctx.op("Mul", ctx.scalar(x, 2), x)
        weighted = ctx.op("Mul", slope, d[0])
        return [
            ctx.tape.op(
                "ReduceSum", [weighted, *node.inputs[1:]], attributes=dict(node.attributes)
            )
        ]
    elif op in ("LogSoftmax", "Softmax"):
        y = node.outputs[0]
        axis = ctx.constant([_int(node, "axis", -1)])
        probability = y if op == "Softmax" else ctx.op("Exp", y)
        weighted = ctx.op("Mul", d[0], probability)
        mean = ctx.op("ReduceSum", weighted, axis, keepdims=1)
        centered = ctx.op("Sub", d[0], mean)
        return [ctx.op("Mul", y, centered) if op == "Softmax" else centered]
    raise NotImplementedError(
        f"No JVP rule for {op} in the reductions family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero tangents."
    )


@_builtin("vjp", OPS)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    op, g = node.op_type, gs[0]
    if op in ("ReduceMean", "ReduceSum"):
        return {0: reduction_vjp(ctx, node, g)}
    elif op in ("ReduceL1", "ReduceSumSquare"):
        x = _input(node)
        slope = ctx.op("Sign", x) if op == "ReduceL1" else ctx.op("Mul", ctx.scalar(x, 2), x)
        return {0: ctx.op("Mul", reduction_vjp(ctx, node, g), slope)}
    elif op in ("LogSoftmax", "Softmax"):
        y = node.outputs[0]
        axis_tensor = ctx.constant([_int(node, "axis", -1)])
        if op == "Softmax":
            weighted = ctx.op("Mul", g, y)
            mean = ctx.op("ReduceSum", weighted, axis_tensor, keepdims=1)
            return {0: ctx.op("Mul", y, ctx.op("Sub", g, mean))}
        total = ctx.op("ReduceSum", g, axis_tensor, keepdims=1)
        return {0: ctx.op("Sub", g, ctx.op("Mul", ctx.op("Exp", y), total))}
    raise NotImplementedError(
        f"No VJP rule for {op} in the reductions family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero cotangents."
    )
