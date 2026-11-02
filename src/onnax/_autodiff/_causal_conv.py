# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Direct products for the depthwise temporal convolution used by Gated DeltaNet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _FLOATS, _input, _rank
from onnax._autodiff._context import _Context
from onnax._rules import _BUILTINS


class _CausalConv:
    def __init__(self, ctx: _Context, node: ir.Node) -> None:
        if len(node.inputs) != 4 or len(node.outputs) != 2:
            raise ValueError("CausalConvWithState requires four inputs and two outputs")
        self.ctx = ctx
        self.x, self.weight, self.bias, self.state = [_input(node, i) for i in range(4)]
        if (
            any(_rank(x) != 3 for x in (self.x, self.weight, self.state))
            or _rank(self.bias) != 1
        ):
            raise NotImplementedError(
                "CausalConvWithState products require one-dimensional depthwise convolution"
            )
        if self.x.dtype not in _FLOATS or any(
            x.dtype != self.x.dtype for x in (self.weight, self.bias, self.state)
        ):
            raise TypeError(
                "CausalConvWithState operands require matching FLOAT or DOUBLE types"
            )
        assert self.weight.shape is not None
        channels, depth, kernel = self.weight.shape
        if (
            not isinstance(channels, int)
            or channels < 1
            or depth != 1
            or not isinstance(kernel, int)
            or kernel < 1
        ):
            raise ValueError(
                "CausalConvWithState requires static positive depthwise channels and kernel size"
            )
        self.channels, self.kernel = channels, kernel
        assert (
            self.x.shape is not None
            and self.state.shape is not None
            and self.bias.shape is not None
        )
        if (
            self.x.shape[1] != channels
            or self.state.shape[1] != channels
            or self.state.shape[2] != kernel - 1
            or self.bias.shape[0] != channels
        ):
            raise ValueError(
                "CausalConvWithState channel and history dimensions disagree with its kernel"
            )
        if set(node.attributes) - {"activation"}:
            raise NotImplementedError(
                "CausalConvWithState products accept the activation attribute"
            )
        activation = (
            node.attributes["activation"].as_string()
            if "activation" in node.attributes
            else "silu"
        )
        if activation not in ("silu", "swish", "none"):
            raise NotImplementedError(
                "CausalConvWithState activation must be silu, swish or none"
            )
        self.combined = ctx.op("Concat", self.state, self.x, axis=2)
        self.length = ctx.op("Shape", self.x, start=2, end=3)
        self.slope = None
        if activation != "none":
            # Match the source function: bias is added after the bias-free Conv.
            preactivation = ctx.op(
                "Add", self.conv(self.combined, self.weight), self.reshape_bias(self.bias)
            )
            sigma = ctx.op("Sigmoid", preactivation)
            one = ctx.scalar(self.x, 1)
            self.slope = ctx.op(
                "Mul",
                sigma,
                ctx.op("Add", one, ctx.op("Mul", preactivation, ctx.op("Sub", one, sigma))),
            )

    def conv(self, x: ir.Value, weight: ir.Value) -> ir.Value:
        return self.ctx.op("Conv", x, weight, group=self.channels, kernel_shape=[self.kernel])

    def reshape_bias(self, bias: ir.Value) -> ir.Value:
        return self.ctx.reshape(bias, self.ctx.constant([1, -1, 1]))

    def slice(self, x: ir.Value, start: ir.Value, end: ir.Value) -> ir.Value:
        return self.ctx.op("Slice", x, start, end, self.ctx.constant([2]))

    def end(self) -> ir.Value:
        return self.ctx.op("Add", self.length, self.ctx.constant([self.kernel - 1]))

    def activate(self, value: ir.Value) -> ir.Value:
        return value if self.slope is None else self.ctx.op("Mul", value, self.slope)


@_BUILTINS.vjp("com.microsoft", "CausalConvWithState", since_version=1, until_version=1)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    a = _CausalConv(ctx, node)
    dpre = a.activate(gs[0])
    result = {}
    if any(i in indices for i in (0, 3)):
        convolution = ctx.op(
            "ConvTranspose", dpre, a.weight, group=a.channels, kernel_shape=[a.kernel]
        )
        pads = ctx.op(
            "Concat", ctx.constant([0, 0]), a.length, ctx.constant([0, 0, 0]), axis=0
        )
        combined = ctx.op("Add", convolution, ctx.op("Pad", gs[1], pads))
        if 0 in indices:
            result[0] = a.slice(combined, ctx.constant([a.kernel - 1]), a.end())
        if 3 in indices:
            result[3] = a.slice(combined, ctx.constant([0]), ctx.constant([a.kernel - 1]))
    if 1 in indices:
        taps = []
        for offset in range(a.kernel):
            window = a.slice(
                a.combined,
                ctx.constant([offset]),
                ctx.op("Add", a.length, ctx.constant([offset])),
            )
            column = ctx.op(
                "ReduceSum", ctx.op("Mul", dpre, window), ctx.constant([0, 2]), keepdims=0
            )
            taps.append(ctx.op("Unsqueeze", column, ctx.constant([1, 2])))
        result[1] = ctx.op("Concat", *taps, axis=2)
    if 2 in indices:
        result[2] = ctx.op("ReduceSum", dpre, ctx.constant([0, 2]), keepdims=0)
    return result


@_BUILTINS.jvp("com.microsoft", "CausalConvWithState", since_version=1, until_version=1)
def jvp(ctx: _Context, node: ir.Node, tangents: Mapping[int, ir.Value]) -> Sequence[ir.Value]:
    a = _CausalConv(ctx, node)
    pieces = []
    combined = None
    if any(i in tangents for i in (0, 3)):
        dx = tangents[0] if 0 in tangents else ctx.like(a.x)
        ds = tangents[3] if 3 in tangents else ctx.like(a.state)
        combined = ctx.op("Concat", ds, dx, axis=2)
        pieces.append(a.conv(combined, a.weight))
    if 1 in tangents:
        pieces.append(a.conv(a.combined, tangents[1]))
    if 2 in tangents:
        pieces.append(a.reshape_bias(tangents[2]))
    # Broadcasting a bias-only tangent must still produce the full output shape.
    differential = ctx.op("Add", ctx.like(node.outputs[0]), ctx.sum(pieces))
    state = ctx.like(a.state) if combined is None else a.slice(combined, a.length, a.end())
    return (a.activate(differential), state)
