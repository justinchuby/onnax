# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Variadic extrema with equal-share tie conventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _input
from onnax._autodiff._context import _Context
from onnax._rules import _builtin

OPS = frozenset(("Max", "Min"))


def _winners(ctx: _Context, node: ir.Node) -> tuple[list[ir.Value], ir.Value]:
    y = node.outputs[0]
    masks = [ctx.op("Equal", _input(node, i), y) for i in range(len(node.inputs))]
    assert y.dtype is not None
    # Count all operand occurrences, even inactive or repeated inputs.
    count = ctx.sum([ctx.op("Cast", mask, to=y.dtype) for mask in masks])
    return masks, count


@_builtin("jvp", OPS)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    masks, count = _winners(ctx, node)
    zero = ctx.scalar(_input(node), 0)
    return [
        ctx.sum(
            [
                ctx.op("Where", masks[i], ctx.op("Div", tangent, count), zero)
                for i, tangent in d.items()
            ]
        )
    ]


@_builtin("vjp", OPS)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    masks, count = _winners(ctx, node)
    shared = ctx.op("Div", gs[0], count)
    zero = ctx.scalar(_input(node), 0)
    return {
        i: ctx.unbroadcast(
            ctx.op("Where", masks[i], shared, zero),
            _input(node, i),
            broadcasted=node.outputs[0],
        )
        for i in indices
    }
