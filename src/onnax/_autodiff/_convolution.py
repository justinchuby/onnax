# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Convolution products for source-layout frozen and trainable image kernels."""

from itertools import product

from onnax._autodiff._common import _input, _int, _rank
from onnax._rules import _BUILTINS


def _configuration(node):
    x, weight = _input(node), _input(node, 1)
    spatial = _rank(x) - 2
    auto = node.attributes.get("auto_pad")
    if spatial < 1 or (auto is not None and auto.as_string() not in ("NOTSET", "VALID")):
        raise NotImplementedError("Conv products require explicit padding or VALID")
    attrs = {name: value for name, value in node.attributes.items() if name != "auto_pad"}
    strides = attrs["strides"].as_ints() if "strides" in attrs else [1] * spatial
    dilations = attrs["dilations"].as_ints() if "dilations" in attrs else [1] * spatial
    pads = attrs["pads"].as_ints() if "pads" in attrs else [0] * (2 * spatial)
    return x, weight, spatial, attrs, strides, dilations, pads


@_BUILTINS.jvp("", "ConvTranspose", since_version=11, until_version=28)
@_BUILTINS.jvp("", "Conv", since_version=11, until_version=28)
def jvp(ctx, node, tangents):
    x, weight, spatial, attrs, *_ = _configuration(node)
    terms = []
    if 0 in tangents:
        terms.append(ctx.op(node.op_type, tangents[0], weight, **attrs))
    if 1 in tangents:
        terms.append(ctx.op(node.op_type, x, tangents[1], **attrs))
    if 2 in tangents:
        shape = [-1] + [1] * spatial
        terms.append(
            ctx.op(
                "Expand",
                ctx.reshape(tangents[2], ctx.constant(shape)),
                ctx.op("Shape", node.outputs[0]),
            )
        )
    return [ctx.sum(terms)]


def _weight_gradient(ctx, node, gradient, configuration):
    x, weight, spatial, _attrs, strides, dilations, pads = configuration
    if weight.shape is None or not all(
        isinstance(dim, int) and dim > 0 for dim in weight.shape
    ):
        raise NotImplementedError("Conv weight products require static kernel dimensions")
    out_channels, in_channels, *kernel = weight.shape
    groups = _int(node, "group", 1)
    if any(pads):
        x = ctx.op("Pad", x, ctx.constant([0, 0, *pads[:spatial], 0, 0, *pads[spatial:]]))
    gs = ctx.op("Reshape", gradient, ctx.constant([0, groups, out_channels // groups, -1]))
    gs = ctx.reshape(
        ctx.op("Transpose", gs, perm=[1, 2, 0, 3]),
        ctx.constant([groups, out_channels // groups, -1]),
    )
    axes = ctx.constant(list(range(2, spatial + 2)))
    shape = ctx.op("Shape", gradient, start=2)
    extent = ctx.op(
        "Add",
        ctx.op(
            "Mul", ctx.op("Sub", shape, ctx.constant([1] * spatial)), ctx.constant(strides)
        ),
        ctx.constant([1] * spatial),
    )
    pieces = []
    for offset in product(*(range(size) for size in kernel)):
        starts = ctx.constant(
            [index * dilation for index, dilation in zip(offset, dilations, strict=True)]
        )
        patch = ctx.op(
            "Slice", x, starts, ctx.op("Add", starts, extent), axes, ctx.constant(strides)
        )
        patch = ctx.op("Reshape", patch, ctx.constant([0, groups, in_channels, -1]))
        patch = ctx.reshape(
            ctx.op("Transpose", patch, perm=[1, 2, 0, 3]),
            ctx.constant([groups, in_channels, -1]),
        )
        value = ctx.op("MatMul", gs, ctx.op("Transpose", patch, perm=[0, 2, 1]))
        pieces.append(ctx.reshape(value, ctx.constant([out_channels, in_channels, 1])))
    return ctx.reshape(ctx.op("Concat", *pieces, axis=2), ctx.op("Shape", weight))


@_BUILTINS.vjp("", "Conv", since_version=11, until_version=28)
def vjp(ctx, node, gradients, indices):
    configuration = _configuration(node)
    x, weight, spatial, attrs, strides, _dilations, _pads = configuration
    result = {}
    if 0 in indices:
        # Produce the largest possible input extent, then crop the stride tail.
        # This keeps odd spatial sizes dynamic without an output_shape attribute.
        value = ctx.op(
            "ConvTranspose",
            gradients[0],
            weight,
            **attrs,
            output_padding=[stride - 1 for stride in strides],
        )
        result[0] = ctx.op(
            "Slice",
            value,
            ctx.constant([0] * spatial),
            ctx.op("Shape", x, start=2),
            ctx.constant(list(range(2, spatial + 2))),
        )
    if 1 in indices:
        result[1] = _weight_gradient(ctx, node, gradients[0], configuration)
    if 2 in indices:
        result[2] = ctx.op(
            "ReduceSum", gradients[0], ctx.constant([0, *range(2, spatial + 2)]), keepdims=0
        )
    return result
