# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Analytic rules that retain fused normalization and activation primals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _float, _input, _int, _rank
from onnax._autodiff._context import _Context
from onnax._rules import _BUILTINS


def _statistics(ctx, node):
    x = _input(node)
    axis = _int(node, "axis", -1)
    if not -_rank(x) <= axis < _rank(x):
        raise ValueError("Normalization axis is outside the input rank")
    axis %= _rank(x)
    axes = ctx.constant(list(range(axis, _rank(x))))
    stash = ir.DataType(_int(node, "stash_type", 1))
    if stash not in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
        raise NotImplementedError(
            "Normalization derivatives require FLOAT or DOUBLE stash_type"
        )
    work = x if x.dtype == stash else ctx.op("Cast", x, to=stash)
    if node.op_type == "LayerNormalization":
        mean = (
            node.outputs[1]
            if len(node.outputs) > 1
            else ctx.op("ReduceMean", work, axes, keepdims=1)
        )
        centered = ctx.op("Sub", work, mean)
    else:
        mean, centered = None, work
    if node.op_type == "LayerNormalization" and len(node.outputs) > 2:
        inverse = node.outputs[2]
    else:
        variance = ctx.op("ReduceMean", ctx.op("Mul", centered, centered), axes, keepdims=1)
        epsilon = ctx.constant(_float(node, "epsilon", 1e-5), stash)
        inverse = ctx.op("Reciprocal", ctx.op("Sqrt", ctx.op("Add", variance, epsilon)))
    normalized = ctx.op("Mul", centered, inverse)
    return axes, work, mean, normalized, inverse, stash


def _cast(ctx, value, dtype):
    return ctx.op("Cast", value, to=dtype)


@_BUILTINS.jvp("", "RMSNormalization", since_version=23, until_version=28)
@_BUILTINS.jvp("", "LayerNormalization", since_version=18, until_version=28)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    x, scale = _input(node), _input(node, 1)
    axes, _work, mean, normalized, inverse, stash = _statistics(ctx, node)
    pieces = []
    dx = _cast(ctx, d[0], stash) if 0 in d else _cast(ctx, ctx.like(x), stash)
    dmean = ctx.op("ReduceMean", dx, axes, keepdims=1)
    dc = ctx.op("Sub", dx, dmean) if mean is not None else dx
    projection = ctx.op("ReduceMean", ctx.op("Mul", normalized, dc), axes, keepdims=1)
    dn = ctx.op("Mul", inverse, ctx.op("Sub", dc, ctx.op("Mul", normalized, projection)))
    if 0 in d:
        pieces.append(ctx.op("Mul", _cast(ctx, dn, x.dtype), scale))
    if 1 in d:
        pieces.append(ctx.op("Mul", _cast(ctx, normalized, x.dtype), d[1]))
    if 2 in d:
        pieces.append(d[2])
    result = [ctx.sum(pieces)]
    if len(node.outputs) > 1:
        result.append(dmean)
    if len(node.outputs) > 2:
        result.append(
            ctx.op("Neg", ctx.op("Mul", ctx.op("Mul", inverse, inverse), projection))
        )
    return result


@_BUILTINS.vjp("", "RMSNormalization", since_version=23, until_version=28)
@_BUILTINS.vjp("", "LayerNormalization", since_version=18, until_version=28)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    x, scale = _input(node), _input(node, 1)
    axes, work, mean, normalized, inverse, stash = _statistics(ctx, node)
    result = {}
    if 0 in indices:
        weighted = _cast(ctx, ctx.op("Mul", gs[0], scale), stash)
        projection = ctx.op(
            "ReduceMean", ctx.op("Mul", weighted, normalized), axes, keepdims=1
        )
        dx = ctx.op("Sub", weighted, ctx.op("Mul", normalized, projection))
        if mean is not None:
            dx = ctx.op("Sub", dx, ctx.op("ReduceMean", weighted, axes, keepdims=1))
        dx = ctx.op("Mul", inverse, dx)
        if len(gs) > 1:
            count = ctx.op(
                "Div",
                ctx.op("Cast", ctx.op("Size", work), to=stash),
                ctx.op("Cast", ctx.op("Size", mean), to=stash),
            )
            dx = ctx.op("Add", dx, ctx.op("Div", gs[1], count))
        if len(gs) > 2:
            extra = ctx.op(
                "Div",
                ctx.op(
                    "Mul", gs[2], ctx.op("Mul", normalized, ctx.op("Mul", inverse, inverse))
                ),
                count,
            )
            dx = ctx.op("Sub", dx, extra)
        result[0] = _cast(ctx, dx, x.dtype)
    if 1 in indices:
        result[1] = ctx.unbroadcast(
            ctx.op("Mul", gs[0], _cast(ctx, normalized, x.dtype)), scale, broadcasted=x
        )
    if 2 in indices:
        result[2] = ctx.unbroadcast(gs[0], _input(node, 2), broadcasted=x)
    return result


def _swish_slope(ctx, node):
    x = _input(node)
    z = ctx.op("Mul", x, ctx.scalar(x, _float(node, "alpha", 1)))
    positive = ctx.op("GreaterOrEqual", z, ctx.scalar(x, 0))
    magnitude = ctx.op("Where", positive, z, ctx.op("Neg", z))
    tail = ctx.op("Exp", ctx.op("Neg", magnitude))
    denominator = ctx.op("Add", ctx.scalar(x, 1), tail)
    probability = ctx.op(
        "Where",
        positive,
        ctx.op("Div", ctx.scalar(x, 1), denominator),
        ctx.op("Div", tail, denominator),
    )
    return ctx.op(
        "Add",
        probability,
        ctx.op(
            "Mul", ctx.op("Mul", z, probability), ctx.op("Sub", ctx.scalar(x, 1), probability)
        ),
    )


@_BUILTINS.jvp("", "Swish", since_version=24, until_version=28)
def swish_jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    return [ctx.op("Mul", d[0], _swish_slope(ctx, node))]


@_BUILTINS.vjp("", "Swish", since_version=24, until_version=28)
def swish_vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    return {0: ctx.op("Mul", gs[0], _swish_slope(ctx, node))}
