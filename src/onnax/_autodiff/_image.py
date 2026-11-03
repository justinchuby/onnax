# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Products for channelwise GroupNorm and integer nearest-neighbor upsampling."""

import numpy as np
import onnx_ir as ir

from onnax._autodiff._common import _float, _input, _int, _rank
from onnax._rules import _BUILTINS


def _group(ctx, node):
    x = _input(node)
    group = _int(node, "num_groups", 1)
    shape = ctx.constant([0, group, -1])
    stash = ir.DataType(_int(node, "stash_type", 1))
    if stash not in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
        raise NotImplementedError(
            "GroupNormalization products require FLOAT or DOUBLE stash_type"
        )
    work = ctx.op("Reshape", ctx.op("Cast", x, to=stash), shape)
    axes = ctx.constant([2])
    centered = ctx.op("Sub", work, ctx.op("ReduceMean", work, axes, keepdims=1))
    variance = ctx.op("ReduceMean", ctx.op("Mul", centered, centered), axes, keepdims=1)
    inverse = ctx.op(
        "Reciprocal",
        ctx.op(
            "Sqrt", ctx.op("Add", variance, ctx.constant(_float(node, "epsilon", 1e-5), stash))
        ),
    )
    normal = ctx.op("Mul", centered, inverse)
    return x, shape, axes, normal, inverse, stash


def _channel(ctx, value, rank):
    return ctx.reshape(value, ctx.constant([-1] + [1] * (rank - 2)))


@_BUILTINS.vjp("", "GroupNormalization", since_version=21, until_version=28)
def group_vjp(ctx, node, gradients, indices):
    x, shape, axes, normal, inverse, stash = _group(ctx, node)
    rank = _rank(x)
    result = {}
    if 0 in indices:
        weighted = ctx.op(
            "Reshape",
            ctx.op(
                "Cast",
                ctx.op("Mul", gradients[0], _channel(ctx, _input(node, 1), rank)),
                to=stash,
            ),
            shape,
        )
        projection = ctx.op("ReduceMean", ctx.op("Mul", weighted, normal), axes, keepdims=1)
        centered = ctx.op("Sub", weighted, ctx.op("ReduceMean", weighted, axes, keepdims=1))
        dx = ctx.op("Mul", inverse, ctx.op("Sub", centered, ctx.op("Mul", normal, projection)))
        result[0] = ctx.op("Cast", ctx.reshape(dx, ctx.op("Shape", x)), to=x.dtype)
    reduction = ctx.constant([0, *range(2, rank)])
    if 1 in indices:
        result[1] = ctx.op(
            "ReduceSum",
            ctx.op(
                "Mul",
                gradients[0],
                ctx.op("Cast", ctx.reshape(normal, ctx.op("Shape", x)), to=x.dtype),
            ),
            reduction,
            keepdims=0,
        )
    if 2 in indices:
        result[2] = ctx.op("ReduceSum", gradients[0], reduction, keepdims=0)
    return result


@_BUILTINS.jvp("", "GroupNormalization", since_version=21, until_version=28)
def group_jvp(ctx, node, tangents):
    x, shape, axes, normal, inverse, stash = _group(ctx, node)
    rank = _rank(x)
    terms = []
    if 0 in tangents:
        dx = ctx.op("Reshape", ctx.op("Cast", tangents[0], to=stash), shape)
        centered = ctx.op("Sub", dx, ctx.op("ReduceMean", dx, axes, keepdims=1))
        projection = ctx.op("ReduceMean", ctx.op("Mul", normal, dx), axes, keepdims=1)
        dn = ctx.op("Mul", inverse, ctx.op("Sub", centered, ctx.op("Mul", normal, projection)))
        terms.append(
            ctx.op(
                "Mul",
                ctx.op("Cast", ctx.reshape(dn, ctx.op("Shape", x)), to=x.dtype),
                _channel(ctx, _input(node, 1), rank),
            )
        )
    if 1 in tangents:
        terms.append(
            ctx.op(
                "Mul",
                ctx.op("Cast", ctx.reshape(normal, ctx.op("Shape", x)), to=x.dtype),
                _channel(ctx, tangents[1], rank),
            )
        )
    if 2 in tangents:
        terms.append(ctx.op("Expand", _channel(ctx, tangents[2], rank), ctx.op("Shape", x)))
    return [ctx.sum(terms)]


def _resize(node):
    def attribute(name, default):
        value = node.attributes.get(name)
        return value.as_string() if value is not None else default

    if attribute("mode", "nearest") != "nearest" or (
        attribute("coordinate_transformation_mode", "half_pixel"),
        attribute("nearest_mode", "round_prefer_floor"),
    ) not in (("half_pixel", "round_prefer_floor"), ("asymmetric", "floor")):
        raise NotImplementedError(
            "Resize products require integer-repeat nearest-neighbor coordinates"
        )
    value = _input(node, 2)
    scales = value.const_value
    producer = value.producer()
    if scales is not None:
        factors = scales.numpy()
    elif (
        producer is not None
        and not producer.domain
        and producer.op_type == "Constant"
        and "value_floats" in producer.attributes
    ):
        factors = np.asarray(producer.attributes["value_floats"].as_floats())
    elif (
        producer is not None
        and not producer.domain
        and producer.op_type == "Constant"
        and "value" in producer.attributes
    ):
        factors = producer.attributes["value"].as_tensor().numpy()
    else:
        raise NotImplementedError("Resize products require constant integer scales")
    if factors.shape != (_rank(_input(node)),) or not np.all(
        np.isfinite(factors) & (factors >= 1) & (factors == np.floor(factors))
    ):
        raise NotImplementedError(
            "Resize products require one positive integer scale per axis"
        )
    if "axes" in node.attributes or (len(node.inputs) > 3 and node.inputs[3] is not None):
        raise NotImplementedError("Resize products use full-rank scales without sizes")
    return [int(value) for value in factors]


@_BUILTINS.vjp("", "Resize", since_version=11, until_version=28)
def resize_vjp(ctx, node, gradients, indices):
    if set(indices) != {0}:
        raise NotImplementedError(
            "Resize products differentiate data with fixed scale controls"
        )
    factors = _resize(node)
    shape: list[ir.Value] = []
    for axis, factor in enumerate(factors):
        shape.extend(
            (ctx.op("Shape", _input(node), start=axis, end=axis + 1), ctx.constant([factor]))
        )
    grouped = ctx.reshape(gradients[0], ctx.op("Concat", *shape, axis=0))
    return {
        0: ctx.op(
            "ReduceSum", grouped, ctx.constant(list(range(1, 2 * len(factors), 2))), keepdims=0
        )
    }


@_BUILTINS.jvp("", "Resize", since_version=11, until_version=28)
def resize_jvp(ctx, node, tangents):
    if set(tangents) != {0}:
        raise NotImplementedError(
            "Resize products differentiate data with fixed scale controls"
        )
    _resize(node)
    return [ctx.op("Resize", tangents[0], *node.inputs[1:], **node.attributes)]
