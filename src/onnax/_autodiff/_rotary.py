# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""RoPE products, retaining fused rotations for data tangents and adjoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _input, _int, _rank
from onnax._autodiff._context import _Context
from onnax._rules import _BUILTINS


def _rotate(ctx, node, x, cos, sin):
    # The adjoint has the primal layout even when upstream shape inference loses
    # the static projection width through broadcast reductions.
    primal = _input(node)
    assert primal.shape is not None
    x = ctx.reshape(x, ctx.op("Shape", primal))
    x.dtype, x.shape = primal.dtype, ir.Shape(primal.shape)
    return ctx.tape.op(
        "RotaryEmbedding", [x, cos, sin, *node.inputs[3:]], dict(node.attributes)
    )


class _Rotary:
    def __init__(self, ctx, node):
        self.ctx, self.node = ctx, node
        self.x, self.cos, self.sin = [_input(node, i) for i in range(3)]
        self.rank = _rank(self.x)
        if self.rank not in (3, 4):
            raise ValueError("RotaryEmbedding expects rank 3 or 4")
        if self.rank == 3:
            heads = _int(node, "num_heads", 0)
            if heads <= 0:
                raise ValueError("Rank-3 RotaryEmbedding requires positive num_heads")
            self.shaped = ctx.op("Reshape", self.x, ctx.constant([0, 0, heads, -1]))
        else:
            self.shaped = ctx.op("Transpose", self.x, perm=[0, 2, 1, 3])
        width = _int(node, "rotary_embedding_dim", 0)
        self.width = (
            ctx.constant([width])
            if width
            else ctx.op("Gather", ctx.op("Shape", self.shaped), ctx.constant([-1]), axis=0)
        )
        self.half = ctx.op("Div", self.width, ctx.constant([2]))
        self.interleaved = _int(node, "interleaved", 0)
        self.positions = node.inputs[3] if len(node.inputs) > 3 else None

    def slice(self, x, start, end, step=1):
        return self.ctx.op(
            "Slice", x, start, end, self.ctx.constant([-1]), self.ctx.constant([step])
        )

    def parts(self, x):
        if self.interleaved:
            return self.slice(x, self.ctx.constant([0]), self.width, 2), self.slice(
                x, self.ctx.constant([1]), self.width, 2
            )
        return self.slice(x, self.ctx.constant([0]), self.half), self.slice(
            x, self.half, self.width
        )

    def shaped_value(self, x):
        if self.rank == 3:
            return self.ctx.reshape(x, self.ctx.op("Shape", self.shaped))
        return self.ctx.op("Transpose", x, perm=[0, 2, 1, 3])

    def cache(self, value):
        if self.positions is not None:
            value = self.ctx.op("Gather", value, self.positions, axis=0)
        return self.ctx.op("Unsqueeze", value, self.ctx.constant([2]))

    def cache_gradient(self, value, primal):
        value = self.ctx.op("ReduceSum", value, self.ctx.constant([2]), keepdims=0)
        if self.positions is None:
            return self.ctx.unbroadcast(value, primal)
        indices = self.ctx.op("Unsqueeze", self.positions, self.ctx.constant([-1]))
        return self.ctx.op("ScatterND", self.ctx.like(primal), indices, value, reduction="add")

    def restore(self, first, second):
        if self.interleaved:
            pair = self.ctx.op(
                "Concat",
                self.ctx.op("Unsqueeze", first, self.ctx.constant([-1])),
                self.ctx.op("Unsqueeze", second, self.ctx.constant([-1])),
                axis=-1,
            )
            shape = self.ctx.op(
                "Concat", self.ctx.op("Shape", first, end=3), self.width, axis=0
            )
            rotated = self.ctx.reshape(pair, shape)
        else:
            rotated = self.ctx.op("Concat", first, second, axis=-1)
        tail = self.slice(self.shaped, self.width, self.ctx.constant([9223372036854775807]))
        output = self.ctx.op(
            "Concat",
            rotated,
            self.ctx.op(
                "ConstantOfShape",
                self.ctx.op("Shape", tail),
                value=ir.tensor([0], dtype=self.x.dtype),
            ),
            axis=-1,
        )
        if self.rank == 3:
            return self.ctx.reshape(output, self.ctx.op("Shape", self.x))
        return self.ctx.op("Transpose", output, perm=[0, 2, 1, 3])


@_BUILTINS.jvp("", "RotaryEmbedding", since_version=23, until_version=28)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    pieces = []
    if 0 in d:
        pieces.append(_rotate(ctx, node, d[0], _input(node, 1), _input(node, 2)))
    if 1 in d or 2 in d:
        a = _Rotary(ctx, node)
        x1, x2 = a.parts(a.shaped)
        real, imaginary = [], []
        if 1 in d:
            dc = a.cache(d[1])
            real.append(ctx.op("Mul", dc, x1))
            imaginary.append(ctx.op("Mul", dc, x2))
        if 2 in d:
            ds = a.cache(d[2])
            real.append(ctx.op("Neg", ctx.op("Mul", ds, x2)))
            imaginary.append(ctx.op("Mul", ds, x1))
        pieces.append(a.restore(ctx.sum(real), ctx.sum(imaginary)))
    return [ctx.sum(pieces)]


@_BUILTINS.vjp("", "RotaryEmbedding", since_version=23, until_version=28)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    result = {}
    if 0 in indices:
        result[0] = _rotate(ctx, node, gs[0], _input(node, 1), ctx.op("Neg", _input(node, 2)))
    if 1 in indices or 2 in indices:
        a = _Rotary(ctx, node)
        x1, x2 = a.parts(a.shaped)
        g1, g2 = a.parts(a.shaped_value(gs[0]))
        if 1 in indices:
            result[1] = a.cache_gradient(
                ctx.op("Add", ctx.op("Mul", g1, x1), ctx.op("Mul", g2, x2)), a.cos
            )
        if 2 in indices:
            result[2] = a.cache_gradient(
                ctx.op("Sub", ctx.op("Mul", g2, x1), ctx.op("Mul", g1, x2)), a.sin
            )
    return result
