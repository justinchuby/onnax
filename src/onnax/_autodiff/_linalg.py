# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Forward and reverse rules for linalg operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _float, _input, _int, _rank
from onnax._autodiff._context import _Context
from onnax._diagnostics import _value_context
from onnax._rules import _builtin

OPS = frozenset(("Gemm", "MatMul"))


def matmul_vjp(
    ctx: _Context, node: ir.Node, gradient: ir.Value, indices: Sequence[int]
) -> dict[int, ir.Value]:
    a, b = _input(node), _input(node, 1)
    ra, rb = _rank(a), _rank(b)
    if ra == 0 or rb == 0:
        raise ValueError(
            "MatMul inputs must have rank at least one; "
            f"got {_value_context(a)} and {_value_context(b)}. "
            "Supply vectors/matrices; use Mul for scalar multiplication."
        )
    av = ctx.op("Unsqueeze", a, ctx.constant([0])) if ra == 1 else a
    bv = ctx.op("Unsqueeze", b, ctx.constant([-1])) if rb == 1 else b
    g = gradient
    if rb == 1:
        g = ctx.op("Unsqueeze", g, ctx.constant([-1]))
    if ra == 1:
        g = ctx.op("Unsqueeze", g, ctx.constant([-2]))
    result = {}
    # Matrix axes never broadcast. When only one operand has batch axes,
    # its adjoint has the exact input shape and the other sums those axes.
    if 0 in indices:
        ga = ctx.op("MatMul", g, ctx.transpose(bv, max(rb, 2)))
        if ra == 1:
            ga = ctx.op("Squeeze", ga, ctx.constant([-2]))
        if rb <= 2:
            result[0] = ga
        elif ra <= 2:
            result[0] = ctx.op("ReduceSum", ga, ctx.constant(list(range(rb - 2))), keepdims=0)
        else:
            result[0] = ctx.unbroadcast(ga, a)
    if 1 in indices:
        gb = ctx.op("MatMul", ctx.transpose(av, max(ra, 2)), g)
        if rb == 1:
            gb = ctx.op("Squeeze", gb, ctx.constant([-1]))
        if ra <= 2:
            result[1] = gb
        elif rb <= 2:
            result[1] = ctx.op("ReduceSum", gb, ctx.constant(list(range(ra - 2))), keepdims=0)
        else:
            result[1] = ctx.unbroadcast(gb, b)
    return result


def gemm(
    ctx: _Context,
    node: ir.Node,
    derivatives: Mapping[int, ir.Value],
    reverse: bool,
    indices: Sequence[int] = (),
) -> dict[int, ir.Value]:
    a, b = _input(node), _input(node, 1)
    ta, tb = _int(node, "transA", 0), _int(node, "transB", 0)
    alpha = _float(node, "alpha", 1)
    beta = _float(node, "beta", 1)
    scalars: dict[float, ir.Value] = {}

    def scale(value: ir.Value, coefficient: float) -> ir.Value:
        if coefficient == 1:
            return value
        if coefficient not in scalars:
            scalars[coefficient] = ctx.scalar(a, coefficient)
        return ctx.op("Mul", value, scalars[coefficient])

    if reverse:
        g = derivatives[0]
        result = {}
        if 0 in indices:
            ga = ctx.op("MatMul", g, b if tb else ctx.transpose(b, 2))
            result[0] = scale(ctx.transpose(ga, 2) if ta else ga, alpha)
        if 1 in indices:
            gb = ctx.op("MatMul", a if ta else ctx.transpose(a, 2), g)
            result[1] = scale(ctx.transpose(gb, 2) if tb else gb, alpha)
        if 2 in indices:
            result[2] = ctx.unbroadcast(
                scale(g, beta), _input(node, 2), broadcasted=node.outputs[0]
            )
        return result
    terms = []
    if 0 in derivatives:
        da = ctx.transpose(derivatives[0], 2) if ta else derivatives[0]
        bv = ctx.transpose(b, 2) if tb else b
        terms.append(scale(ctx.op("MatMul", da, bv), alpha))
    if 1 in derivatives:
        db = ctx.transpose(derivatives[1], 2) if tb else derivatives[1]
        av = ctx.transpose(a, 2) if ta else a
        terms.append(scale(ctx.op("MatMul", av, db), alpha))
    if 2 in derivatives:
        terms.append(scale(derivatives[2], beta))
    # A bias-only direction still has the full Gemm output shape.
    return {0: ctx.op("Expand", ctx.sum(terms), ctx.op("Shape", node.outputs[0]))}


@_builtin("jvp", OPS)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    op = node.op_type
    if op in ("MatMul",):
        terms = []
        if 0 in d:
            terms.append(ctx.op("MatMul", d[0], _input(node, 1)))
        if 1 in d:
            terms.append(ctx.op("MatMul", _input(node), d[1]))
        return [ctx.sum(terms)]
    elif op in ("Gemm",):
        return [gemm(ctx, node, d, reverse=False)[0]]
    raise NotImplementedError(
        f"No JVP rule for {op} in the linear algebra family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero tangents."
    )


@_builtin("vjp", OPS)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    op, g = node.op_type, gs[0]
    if op in ("MatMul",):
        return matmul_vjp(ctx, node, g, indices)
    elif op in ("Gemm",):
        return gemm(ctx, node, {0: g}, reverse=True, indices=indices)
    raise NotImplementedError(
        f"No VJP rule for {op} in the linear algebra family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero cotangents."
    )
