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
    (
        "LogSoftmax",
        "ReduceL1",
        "ReduceLogSum",
        "ReduceMean",
        "ReduceSum",
        "ReduceSumSquare",
        "Softmax",
    )
)


def _pivot_mask(ctx: _Context, node: ir.Node) -> ir.Value:
    """Select one maximum, including for ties and symbolic axis lengths."""
    x = _input(node)
    axis = _int(node, "axis", -1)
    index = ctx.op("ArgMax", x, axis=axis, keepdims=1)
    updates = ctx.op(
        "ConstantOfShape", ctx.op("Shape", index), value=ir.tensor([1], dtype=x.dtype)
    )
    mask = ctx.op("ScatterElements", ctx.like(x), index, updates, axis=axis)
    return ctx.op("Cast", mask, to=ir.DataType.BOOL)


def _centered(
    ctx: _Context, value: ir.Value, mask: ir.Value, axis: ir.Value, zero: ir.Value
) -> ir.Value:
    pivot = ctx.op("ReduceSum", ctx.op("Where", mask, value, zero), axis, keepdims=1)
    return ctx.op("Where", mask, zero, ctx.op("Sub", value, pivot))


def _conserve(
    ctx: _Context, value: ir.Value, mask: ir.Value, axis: ir.Value, zero: ir.Value
) -> ir.Value:
    others = ctx.op("Where", mask, zero, value)
    pivot = ctx.op("Neg", ctx.op("ReduceSum", others, axis, keepdims=1))
    return ctx.op("Where", mask, pivot, others)


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
        # Restore removed axes directly. Empty runtime axes leave a scalar or
        # identity reduction, either of which still expands to the primal shape.
        g = ctx.op("Unsqueeze", g, axes)
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
    elif op == "ReduceLogSum":
        reduced = ctx.tape.op(
            "ReduceSum", [d[0], *node.inputs[1:]], attributes=dict(node.attributes)
        )
        return [ctx.op("Mul", reduced, ctx.op("Exp", ctx.op("Neg", node.outputs[0])))]
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
        mask = _pivot_mask(ctx, node)
        differences = _centered(ctx, d[0], mask, axis, ctx.scalar(_input(node), 0))
        mean = ctx.op("ReduceSum", ctx.op("Mul", differences, probability), axis, keepdims=1)
        centered = ctx.op("Sub", differences, mean)
        if op == "Softmax":
            return [
                _conserve(
                    ctx, ctx.op("Mul", y, centered), mask, axis, ctx.scalar(_input(node), 0)
                )
            ]
        return [centered]
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
    elif op == "ReduceLogSum":
        # exp(-log(sum(x))) reuses the primal reduction without reconstructing
        # its potentially very large sum through exp(y).
        scaled = ctx.op("Mul", g, ctx.op("Exp", ctx.op("Neg", node.outputs[0])))
        return {0: reduction_vjp(ctx, node, scaled)}
    elif op in ("ReduceL1", "ReduceSumSquare"):
        x = _input(node)
        slope = ctx.op("Sign", x) if op == "ReduceL1" else ctx.op("Mul", ctx.scalar(x, 2), x)
        return {0: ctx.op("Mul", reduction_vjp(ctx, node, g), slope)}
    elif op in ("LogSoftmax", "Softmax"):
        y = node.outputs[0]
        axis_tensor = ctx.constant([_int(node, "axis", -1)])
        mask = _pivot_mask(ctx, node)
        if op == "Softmax":
            differences = _centered(ctx, g, mask, axis_tensor, ctx.scalar(_input(node), 0))
            weighted = ctx.op("Mul", differences, y)
            mean = ctx.op("ReduceSum", weighted, axis_tensor, keepdims=1)
            derivative = ctx.op("Mul", y, ctx.op("Sub", differences, mean))
        else:
            total = ctx.op("ReduceSum", g, axis_tensor, keepdims=1)
            derivative = ctx.op("Sub", g, ctx.op("Mul", ctx.op("Exp", y), total))
        return {0: _conserve(ctx, derivative, mask, axis_tensor, ctx.scalar(_input(node), 0))}
    raise NotImplementedError(
        f"No VJP rule for {op} in the reductions family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero cotangents."
    )
