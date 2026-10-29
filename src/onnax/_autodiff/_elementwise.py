# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Forward and reverse rules for elementwise operators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import onnx_ir as ir

from ._common import _BINARY, _FLOATS, _float, _input
from ._context import _Context

OPS = frozenset(
    (
        "Abs",
        "Add",
        "Cos",
        "Cosh",
        "Div",
        "Erf",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Log",
        "Mul",
        "Neg",
        "Pow",
        "Reciprocal",
        "Relu",
        "Sigmoid",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Tan",
        "Tanh",
    )
)


def _partials(ctx: _Context, node: ir.Node, indices: Sequence[int]) -> dict[int, ir.Value]:
    """Elementwise local derivatives; do not evaluate unused partials (notably Pow)."""
    op = node.op_type
    x, y = _input(node), node.outputs[0]
    assert x.dtype in _FLOATS
    one = ctx.scalar(x, 1)
    result = {}
    for index in indices:
        if op in ("Identity", "Add"):
            p = one
        elif op == "Sub":
            p = one if index == 0 else ctx.scalar(x, -1)
        elif op == "Mul":
            p = _input(node, 1 - index)
        elif op == "Div":
            b = _input(node, 1)
            p = ctx.op("Div", one, b) if index == 0 else ctx.op("Neg", ctx.op("Div", y, b))
        elif op == "Pow":
            b = _input(node, 1)
            if b.dtype != x.dtype:
                b = ctx.op("Cast", b, to=x.dtype)
            if index == 0:
                # Guard only 0**0, preserving mixed exponent derivatives at b=0
                # for positive bases. A blanket b=0 branch loses that dependence.
                zero = ctx.scalar(x, 0)
                at_origin = ctx.op("And", ctx.op("Equal", x, zero), ctx.op("Equal", b, zero))
                safe_b = ctx.op("Where", at_origin, one, b)
                p = ctx.op("Mul", b, ctx.op("Pow", x, ctx.op("Sub", safe_b, one)))
            else:
                p = ctx.op("Mul", y, ctx.op("Log", x))
        elif op == "Neg":
            p = ctx.scalar(x, -1)
        elif op == "Abs":
            p = ctx.op("Sign", x)
        elif op == "Exp":
            p = y
        elif op == "Log":
            p = ctx.op("Div", one, x)
        elif op == "Sqrt":
            p = ctx.op("Div", ctx.scalar(x, 0.5), y)
        elif op == "Reciprocal":
            p = ctx.op("Neg", ctx.op("Mul", y, y))
        elif op == "Sin":
            p = ctx.op("Cos", x)
        elif op == "Cos":
            p = ctx.op("Neg", ctx.op("Sin", x))
        elif op == "Tan":
            p = ctx.op("Add", one, ctx.op("Mul", y, y))
        elif op == "Sinh":
            p = ctx.op("Cosh", x)
        elif op == "Cosh":
            p = ctx.op("Sinh", x)
        elif op == "Tanh":
            p = ctx.op("Sub", one, ctx.op("Mul", y, y))
        elif op == "Sigmoid":
            p = ctx.op("Mul", y, ctx.op("Sub", one, y))
        elif op in ("Relu", "LeakyRelu"):
            negative = 0.0 if op == "Relu" else _float(node, "alpha", 0.01)
            p = ctx.op(
                "Where",
                ctx.op("Greater", x, ctx.scalar(x, 0)),
                one,
                ctx.scalar(x, negative),
            )
        elif op == "Softplus":
            p = ctx.op("Sigmoid", x)
        elif op == "Softsign":
            denom = ctx.op("Add", one, ctx.op("Abs", x))
            p = ctx.op("Div", one, ctx.op("Mul", denom, denom))
        elif op == "Erf":
            p = ctx.op(
                "Mul",
                ctx.scalar(x, 2 / math.sqrt(math.pi)),
                ctx.op("Exp", ctx.op("Neg", ctx.op("Mul", x, x))),
            )
        else:
            raise NotImplementedError(f"No elementwise derivative for {op}.")
        result[index] = p
    return result


def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    op = node.op_type
    if op in (
        "Abs",
        "Add",
        "Cos",
        "Cosh",
        "Div",
        "Erf",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Log",
        "Mul",
        "Neg",
        "Pow",
        "Reciprocal",
        "Relu",
        "Sigmoid",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Tan",
        "Tanh",
    ):
        partials = _partials(ctx, node, tuple(d))
        result = ctx.sum([ctx.op("Mul", d[i], partials[i]) for i in d])
        return [
            ctx.op("Expand", result, ctx.op("Shape", node.outputs[0]))
            if op in _BINARY
            else result
        ]
    raise NotImplementedError(f"No JVP rule for {op}.")


def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    op, g = node.op_type, gs[0]
    if op in (
        "Abs",
        "Add",
        "Cos",
        "Cosh",
        "Div",
        "Erf",
        "Exp",
        "Identity",
        "LeakyRelu",
        "Log",
        "Mul",
        "Neg",
        "Pow",
        "Reciprocal",
        "Relu",
        "Sigmoid",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Tan",
        "Tanh",
    ):
        partials = _partials(ctx, node, indices)
        result = {}
        for i in indices:
            contribution = ctx.op("Mul", g, partials[i])
            result[i] = (
                ctx.unbroadcast(contribution, _input(node, i))
                if op in _BINARY
                else contribution
            )
        return result
    raise NotImplementedError(f"No VJP rule for {op}.")
