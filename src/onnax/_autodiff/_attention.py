# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Cache-free Attention products with shared, grouped Q/K/V derivatives.

The primal stays fused. Portable derivatives use standard ONNX operations and
keep KV heads grouped instead of copying them to every query head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _float, _input, _int, _rank
from onnax._autodiff._context import _Context
from onnax._autodiff._reductions import _centered, _conserve
from onnax._rules import _BUILTINS


class _Attention:
    def __init__(self, ctx, node):
        self.ctx, self.node = ctx, node
        if len(node.outputs) != 1 or any(x is not None for x in node.inputs[4:]):
            raise NotImplementedError(
                "Attention derivatives support cache-free training with only the Y output"
            )
        self.q, self.k, self.v = [_input(node, i) for i in range(3)]
        if self.q.dtype not in (ir.DataType.FLOAT, ir.DataType.DOUBLE) or any(
            x.dtype != self.q.dtype for x in (self.k, self.v)
        ):
            raise NotImplementedError(
                "Attention derivatives require matching FLOAT or DOUBLE Q/K/V tensors"
            )
        self.rank = _rank(self.q)
        if self.rank not in (3, 4) or any(_rank(x) != self.rank for x in (self.k, self.v)):
            raise ValueError("Attention Q/K/V must have matching rank 3 or 4")
        if self.rank == 3:
            self.heads = [_int(node, "q_num_heads", 0), _int(node, "kv_num_heads", 0)]
        else:
            assert (
                self.q.shape is not None
                and self.k.shape is not None
                and self.v.shape is not None
            )
            qh, kh = self.q.shape[1], self.k.shape[1]
            if not isinstance(qh, int) or not isinstance(kh, int):
                raise ValueError("Attention derivatives require static head counts")
            self.heads = [qh, kh]
            if self.v.shape[1] != self.heads[1]:
                raise ValueError("Attention K/V head counts must match")
        if any(not isinstance(h, int) or h <= 0 for h in self.heads):
            raise ValueError(
                "Attention derivatives require known positive query and KV head counts"
            )
        self.hq, self.hk = self.heads
        if self.hq % self.hk:
            raise ValueError("Attention query heads must be divisible by KV heads")
        self.groups = self.hq // self.hk
        self.q4, self.k4, self.v4 = [
            self.to4(x, h)
            for x, h in zip((self.q, self.k, self.v), (self.hq, self.hk, self.hk))
        ]
        self.q5 = self.group(self.q4)
        self.k5 = ctx.op("Unsqueeze", self.k4, ctx.constant([2]))
        self.v5 = ctx.op("Unsqueeze", self.v4, ctx.constant([2]))
        if "scale" in node.attributes:
            scale = ctx.scalar(self.q, _float(node, "scale", 1))
        else:
            width = ctx.op("Cast", self.dim(self.q4, 3), to=self.q.dtype)
            scale = ctx.op("Reciprocal", ctx.op("Sqrt", width))
        # Match ONNX's split scaling, including its rounding in FP32.
        self.root_scale = ctx.op("Sqrt", scale)
        self.qs = ctx.op("Mul", self.q5, self.root_scale)
        self.ks = ctx.op("Mul", self.k5, self.root_scale)
        raw = ctx.op("MatMul", self.qs, ctx.transpose(self.ks, 5))
        scores = self.flatten(raw, self.dim(self.k4, 2))
        cap = _float(node, "softcap", 0)
        self.cap_slope = None
        if cap > 0:
            limited = ctx.op("Tanh", ctx.op("Div", scores, ctx.scalar(self.q, cap)))
            self.cap_slope = ctx.op(
                "Sub", ctx.scalar(self.q, 1), ctx.op("Mul", limited, limited)
            )
            scores = ctx.op("Mul", limited, ctx.scalar(self.q, cap))
        mask = node.inputs[3] if len(node.inputs) > 3 else None
        if mask is not None:
            if not 1 <= _rank(mask) <= 4:
                raise NotImplementedError(
                    "Attention derivatives support masks of rank 1 through 4"
                )
            # Short masks are valid: remaining keys are excluded.
            padding = ctx.op("Sub", self.dim(self.k4, 2), self.dim(mask, -1))
            pads = ctx.op("Concat", ctx.constant([0] * (2 * _rank(mask) - 1)), padding, axis=0)
            if mask.dtype == ir.DataType.BOOL:
                mask = ctx.op("Pad", mask, pads, ctx.constant(False, ir.DataType.BOOL))
                bias = ctx.op(
                    "Where", mask, ctx.scalar(self.q, 0), ctx.scalar(self.q, float("-inf"))
                )
            else:
                bias = ctx.op("Pad", mask, pads, ctx.scalar(self.q, float("-inf")))
            scores = ctx.op("Add", scores, bias)
        left, right = _int(node, "left_window_size", -1), _int(node, "right_window_size", -1)
        if _int(node, "is_causal", 0) or left >= 0 or right >= 0:
            qi = self.positions(self.q4)
            ki = ctx.op("Transpose", self.positions(self.k4), perm=[1, 0])
            constraints = []
            if _int(node, "is_causal", 0):
                constraints.append(ctx.op("LessOrEqual", ki, qi))
            if left >= 0:
                constraints.append(
                    ctx.op("GreaterOrEqual", ki, ctx.op("Sub", qi, ctx.constant(left)))
                )
            if right >= 0:
                constraints.append(
                    ctx.op("LessOrEqual", ki, ctx.op("Add", qi, ctx.constant(right)))
                )
            allowed = constraints[0]
            for constraint in constraints[1:]:
                allowed = ctx.op("And", allowed, constraint)
            scores = ctx.op(
                "Add",
                scores,
                ctx.op(
                    "Where", allowed, ctx.scalar(self.q, 0), ctx.scalar(self.q, float("-inf"))
                ),
            )
        assert self.q.dtype is not None
        precision = _int(node, "softmax_precision", int(self.q.dtype))
        if precision not in (int(ir.DataType.FLOAT), int(ir.DataType.DOUBLE)):
            raise NotImplementedError(
                "Attention derivatives require FLOAT or DOUBLE softmax precision"
            )
        work = ctx.op("Cast", scores, to=precision)
        # Softmax has undefined all--inf rows; Attention defines those rows as zero.
        maximum = ctx.op("ReduceMax", work, ctx.constant([-1]), keepdims=1)
        empty = ctx.op("Equal", maximum, ctx.constant(float("-inf"), ir.DataType(precision)))
        safe = ctx.op("Where", empty, ctx.constant(0, ir.DataType(precision)), work)
        probability = ctx.op("Softmax", safe, axis=-1)
        self.precision = ir.DataType(precision)
        self.p_work = ctx.op("Where", empty, ctx.constant(0, self.precision), probability)
        self.p = ctx.op("Cast", self.p_work, to=self.q.dtype)
        self.p5 = self.group(self.p)

    def dim(self, x, index):
        return self.ctx.op(
            "Gather", self.ctx.op("Shape", x), self.ctx.constant([index]), axis=0
        )

    def to4(self, value, heads):
        if self.rank == 4:
            return value
        shaped = self.ctx.op("Reshape", value, self.ctx.constant([0, 0, heads, -1]))
        return self.ctx.op("Transpose", shaped, perm=[0, 2, 1, 3])

    def restore(self, value, primal):
        if self.rank == 3:
            value = self.ctx.op("Transpose", value, perm=[0, 2, 1, 3])
            value = self.ctx.reshape(value, self.ctx.op("Shape", primal))
        return value

    def group(self, value):
        shape = self.ctx.op(
            "Concat",
            self.dim(value, 0),
            self.ctx.constant([self.hk, self.groups]),
            self.dim(value, 2),
            self.dim(value, 3),
            axis=0,
        )
        return self.ctx.reshape(value, shape)

    def flatten(self, value, width):
        shape = self.ctx.op(
            "Concat",
            self.dim(self.q4, 0),
            self.ctx.constant([self.hq]),
            self.dim(self.q4, 2),
            width,
            axis=0,
        )
        return self.ctx.reshape(value, shape)

    def positions(self, value):
        stop = self.ctx.op("Squeeze", self.dim(value, 2), self.ctx.constant([0]))
        return self.ctx.op(
            "Unsqueeze",
            self.ctx.op("Range", self.ctx.constant(0), stop, self.ctx.constant(1)),
            self.ctx.constant([1]),
        )

    def softmax_product(self, value):
        ctx = self.ctx
        probability = self.p_work
        value = ctx.op("Cast", value, to=self.precision)
        axis = ctx.constant([-1])
        index = ctx.op("ArgMax", probability, axis=-1, keepdims=1)
        ones = ctx.op(
            "ConstantOfShape",
            ctx.op("Shape", index),
            value=ir.tensor([1], dtype=self.precision),
        )
        zeros = ctx.op(
            "ConstantOfShape",
            ctx.op("Shape", probability),
            value=ir.tensor([0], dtype=self.precision),
        )
        pivot = ctx.op(
            "Cast", ctx.op("ScatterElements", zeros, index, ones, axis=-1), to=ir.DataType.BOOL
        )
        zero = ctx.constant(0, self.precision)
        differences = _centered(ctx, value, pivot, axis, zero)
        total = ctx.op("ReduceSum", ctx.op("Mul", probability, differences), axis, keepdims=1)
        derivative = ctx.op("Mul", probability, ctx.op("Sub", differences, total))
        return ctx.op("Cast", _conserve(ctx, derivative, pivot, axis, zero), to=self.q.dtype)

    def reduce_kv(self, value, primal):
        value = self.ctx.op("ReduceSum", value, self.ctx.constant([2]), keepdims=0)
        return self.restore(value, primal)


@_BUILTINS.vjp("", "Attention", since_version=23, until_version=28)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    a = _Attention(ctx, node)
    g5 = a.group(a.to4(gs[0], a.hq))
    result = {}
    if 2 in indices:
        result[2] = a.reduce_kv(ctx.op("MatMul", ctx.transpose(a.p5, 5), g5), a.v)
    if any(i in indices for i in (0, 1, 3)):
        dp = a.flatten(ctx.op("MatMul", g5, ctx.transpose(a.v5, 5)), a.dim(a.k4, 2))
        ds = a.softmax_product(dp)
        if 3 in indices:
            # Undo any mask padding before reducing broadcast axes.
            mask = _input(node, 3)
            sliced = ctx.op(
                "Slice", ds, ctx.constant([0]), a.dim(mask, -1), ctx.constant([-1])
            )
            result[3] = ctx.unbroadcast(sliced, mask)
        if a.cap_slope is not None:
            ds = ctx.op("Mul", ds, a.cap_slope)
        ds5 = a.group(ds)
        if 0 in indices:
            dq = ctx.op("Mul", ctx.op("MatMul", ds5, a.ks), a.root_scale)
            result[0] = a.restore(a.flatten(dq, a.dim(a.q4, 3)), a.q)
        if 1 in indices:
            dk = ctx.op("Mul", ctx.op("MatMul", ctx.transpose(ds5, 5), a.qs), a.root_scale)
            result[1] = a.reduce_kv(dk, a.k)
    return result


@_BUILTINS.jvp("", "Attention", since_version=23, until_version=28)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    a = _Attention(ctx, node)
    score_terms = []
    if 0 in d:
        dq = ctx.op("Mul", a.group(a.to4(d[0], a.hq)), a.root_scale)
        score_terms.append(ctx.op("MatMul", dq, ctx.transpose(a.ks, 5)))
    if 1 in d:
        dk = ctx.op(
            "Mul", ctx.op("Unsqueeze", a.to4(d[1], a.hk), ctx.constant([2])), a.root_scale
        )
        score_terms.append(ctx.op("MatMul", a.qs, ctx.transpose(dk, 5)))
    terms = []
    if score_terms or 3 in d:
        ds = a.flatten(ctx.sum(score_terms), a.dim(a.k4, 2)) if score_terms else ctx.like(a.q)
        if a.cap_slope is not None and score_terms:
            ds = ctx.op("Mul", ds, a.cap_slope)
        if 3 in d:
            mask = d[3]
            pads = ctx.op(
                "Concat",
                ctx.constant([0] * (2 * _rank(_input(node, 3)) - 1)),
                ctx.op("Sub", a.dim(a.k4, 2), a.dim(_input(node, 3), -1)),
                axis=0,
            )
            dm = ctx.op("Pad", mask, pads, ctx.scalar(a.q, 0))
            ds = ctx.op("Add", ds, dm) if score_terms else dm
        terms.append(ctx.op("MatMul", a.group(a.softmax_product(ds)), a.v5))
    if 2 in d:
        dv = ctx.op("Unsqueeze", a.to4(d[2], a.hk), ctx.constant([2]))
        terms.append(ctx.op("MatMul", a.p5, dv))
    output = a.flatten(ctx.sum(terms), a.dim(a.v4, 3))
    return [a.restore(output, node.outputs[0])]
