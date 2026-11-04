# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Forward and reverse rules for layout operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import onnx_ir as ir

from onnax._autodiff._common import _FLOATS, _input, _int, _rank
from onnax._autodiff._context import _Context
from onnax._rules import _builtin

OPS = frozenset(
    (
        "Cast",
        "CastLike",
        "Concat",
        "Expand",
        "Flatten",
        "Gather",
        "Reshape",
        "ScatterND",
        "Split",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
        "Where",
    )
)


def gather_vjp(ctx: _Context, node: ir.Node, g: ir.Value) -> ir.Value:
    x, indices = _input(node), _input(node, 1)
    rank, index_rank = _rank(x), _rank(indices)
    axis = _int(node, "axis", 0) % rank
    perm = [axis, *range(axis), *range(axis + 1, rank)]
    updates_perm = [
        *range(axis, axis + index_rank),
        *range(axis),
        *range(axis + index_rank, rank - 1 + index_rank),
    ]
    updates = ctx.op("Transpose", g, perm=updates_perm)
    zeros = ctx.op("Transpose", ctx.like(x), perm=perm)
    if indices.dtype != ir.DataType.INT64:
        indices = ctx.op("Cast", indices, to=ir.DataType.INT64)
    result = ctx.op(
        "ScatterND",
        zeros,
        ctx.op("Unsqueeze", indices, ctx.constant([-1])),
        updates,
        reduction="add",
    )
    return ctx.op("Transpose", result, perm=np.argsort(perm).tolist())


@_builtin("jvp", OPS)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    op = node.op_type
    if op in (
        "Cast",
        "CastLike",
        "Expand",
        "Flatten",
        "Gather",
        "Reshape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    ):
        return [ctx.tape.op(op, [d[0], *node.inputs[1:]], attributes=dict(node.attributes))]
    elif op in ("Where",):
        return [
            ctx.op(
                "Where",
                _input(node),
                d[1] if 1 in d else ctx.like(_input(node, 1)),
                d[2] if 2 in d else ctx.like(_input(node, 2)),
            )
        ]
    elif op in ("Concat",):
        return [
            ctx.op(
                "Concat",
                *(
                    d[i] if i in d else ctx.like(_input(node, i))
                    for i in range(len(node.inputs))
                ),
                axis=_int(node, "axis", 0),
            )
        ]
    elif op in ("Split",):
        return list(
            ctx.tape.op_multi_out(
                "Split",
                [d[0], *node.inputs[1:]],
                dict(node.attributes),
                num_outputs=len(node.outputs),
            )
        )
    elif op in ("ScatterND",):
        reduction = node.attributes.get("reduction")
        mode = reduction.as_string() if reduction is not None else "none"
        if mode not in ("none", "add"):
            raise NotImplementedError(
                "JVP ScatterND requires reduction='none' or reduction='add'; "
                f"got reduction={mode!r} (node {node.name!r}). "
                "Use a separate derivative for mul/min/max; changing reduction changes the function."
            )
        return [
            ctx.op(
                "ScatterND",
                d[0] if 0 in d else ctx.like(_input(node)),
                _input(node, 1),
                d[2] if 2 in d else ctx.like(_input(node, 2)),
                reduction=mode,
            )
        ]
    raise NotImplementedError(
        f"No JVP rule for {op} in the layout family. "
        "Correct the private registry dispatch or implement and validate the missing "
        "layout rule; unsupported active paths cannot be replaced with zero tangents."
    )


@_builtin("vjp", OPS - {"ScatterND"})
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    op, g = node.op_type, gs[0]
    if op in ("Flatten", "Reshape", "Squeeze", "Unsqueeze"):
        return {0: ctx.reshape(g, ctx.op("Shape", _input(node)))}
    elif op in ("Expand",):
        return {0: ctx.unbroadcast(g, _input(node), broadcasted=node.outputs[0])}
    elif op in ("Transpose",):
        attr = node.attributes.get("perm")
        perm = (
            attr.as_ints() if attr is not None else tuple(reversed(range(_rank(_input(node)))))
        )
        return {0: ctx.op("Transpose", g, perm=np.argsort(perm).tolist())}
    elif op in ("Cast", "CastLike"):
        dtype = _input(node).dtype
        assert dtype in _FLOATS
        return {0: ctx.op("Cast", g, to=dtype)}
    elif op in ("Where",):
        zero = ctx.scalar(node.outputs[0], 0)
        return {
            i: ctx.unbroadcast(
                ctx.op("Where", _input(node), g if i == 1 else zero, zero if i == 1 else g),
                _input(node, i),
                broadcasted=node.outputs[0],
            )
            for i in indices
        }
    elif op in ("Concat",):
        axis = _int(node, "axis", 0)
        sizes = [
            ctx.op("Gather", ctx.op("Shape", _input(node, i)), ctx.constant([axis]), axis=0)
            for i in range(len(node.inputs))
        ]
        parts = ctx.tape.op_multi_out(
            "Split",
            [g, ctx.op("Concat", *sizes, axis=0)],
            {"axis": axis},
            num_outputs=len(node.inputs),
        )
        return {i: parts[i] for i in indices}
    elif op in ("Split",):
        return {0: ctx.op("Concat", *gs, axis=_int(node, "axis", 0))}
    elif op in ("Gather",):
        return {0: gather_vjp(ctx, node, g)}
    raise NotImplementedError(
        f"No VJP rule for {op} in the layout family. "
        "Correct the private registry dispatch or implement and validate the missing "
        "layout rule; unsupported active paths cannot be replaced with zero cotangents."
    )


@_builtin("jvp", frozenset(("Slice",)))
def slice_jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    """Propagate structural slices used by fused-operator derivative graphs."""
    return [ctx.tape.op("Slice", [d[0], *node.inputs[1:]], dict(node.attributes))]


@_builtin("jvp", frozenset(("Pad",)))
def pad_jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    if node.attributes.get_string("mode", "constant") != "constant":
        return (
            [ctx.tape.op("Pad", [d[0], *node.inputs[1:]], dict(node.attributes))]
            if 0 in d
            else [ctx.like(node.outputs[0])]
        )
    x = _input(node)
    # Differentiating a fixed boundary value pads the tangent with zeros.
    value = d[0] if 0 in d else ctx.like(x)
    fill = d[2] if 2 in d else ctx.scalar(x, 0)
    return [
        ctx.tape.op(
            "Pad", [value, _input(node, 1), fill, *node.inputs[3:]], dict(node.attributes)
        )
    ]


@_builtin("vjp", frozenset(("ScatterND",)))
def scatter_nd_vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    """Pull back additive scatter or replacement at unique index tuples."""
    reduction = node.attributes.get_string("reduction", "none")
    if reduction not in ("none", "add"):
        raise NotImplementedError(
            f"VJP ScatterND requires reduction='none' or reduction='add'; got reduction={reduction!r}. "
            "Use a separate derivative for other reductions without changing semantics."
        )
    contributions = {}
    if 0 in indices:
        contributions[0] = (
            gs[0]
            if reduction == "add"
            else ctx.op("ScatterND", gs[0], _input(node, 1), ctx.like(_input(node, 2)))
        )
    if 2 in indices:
        contributions[2] = ctx.op("GatherND", gs[0], _input(node, 1))
    return contributions


@_builtin("vjp", frozenset(("Slice",)))
def slice_vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    """Scatter a sliced cotangent back to its original element positions."""
    x = _input(node)
    rank = _rank(x)
    if rank == 0:
        return {0: gs[0]}
    shape = ctx.op("Shape", x)
    starts, ends = (ctx.op("Cast", _input(node, i), to=ir.DataType.INT64) for i in (1, 2))
    axes = (
        ctx.op("Cast", node.inputs[3], to=ir.DataType.INT64)
        if len(node.inputs) > 3 and node.inputs[3] is not None
        else ctx.op("Range", ctx.constant(0), ctx.op("Size", starts), ctx.constant(1))
    )
    axes = ctx.op("Mod", ctx.op("Add", axes, ctx.constant(rank)), ctx.constant(rank))
    steps = (
        ctx.op("Cast", node.inputs[4], to=ir.DataType.INT64)
        if len(node.inputs) > 4 and node.inputs[4] is not None
        else ctx.op(
            "ConstantOfShape",
            ctx.op("Shape", starts),
            value=ir.tensor([1], dtype=ir.DataType.INT64),
        )
    )
    padded_starts = ctx.op("Concat", starts, ctx.constant([0]), axis=0)
    padded_steps = ctx.op("Concat", steps, ctx.constant([1]), axis=0)
    positions, stride = ctx.constant(0), ctx.constant(1)
    for axis in reversed(range(rank)):
        dim = ctx.op("Gather", shape, ctx.constant(axis), axis=0)
        # A sentinel match supplies the full axis for dimensions not sliced.
        matches = ctx.op(
            "Concat",
            ctx.op("Equal", axes, ctx.constant(axis)),
            ctx.constant([True], ir.DataType.BOOL),
            axis=0,
        )
        slot = ctx.op(
            "ArgMax", ctx.op("Cast", matches, to=ir.DataType.INT64), axis=0, keepdims=1
        )
        padded_ends = ctx.op(
            "Concat", ends, ctx.op("Unsqueeze", dim, ctx.constant([0])), axis=0
        )
        coordinates = ctx.op("Range", ctx.constant(0), dim, ctx.constant(1))
        coordinates = ctx.op(
            "Slice",
            coordinates,
            ctx.op("Gather", padded_starts, slot, axis=0),
            ctx.op("Gather", padded_ends, slot, axis=0),
            ctx.constant([0]),
            ctx.op("Gather", padded_steps, slot, axis=0),
        )
        broadcast_shape = [1] * rank
        broadcast_shape[axis] = -1
        positions = ctx.op(
            "Add",
            positions,
            ctx.op("Mul", ctx.reshape(coordinates, ctx.constant(broadcast_shape)), stride),
        )
        stride = ctx.op("Mul", stride, dim)
    sliced = positions
    scattered = ctx.op(
        "ScatterND",
        ctx.reshape(ctx.like(x), ctx.constant([-1])),
        ctx.reshape(sliced, ctx.constant([-1, 1])),
        ctx.reshape(gs[0], ctx.constant([-1])),
        reduction="add",
    )
    return {0: ctx.reshape(scattered, ctx.op("Shape", x))}


@_builtin("vjp", frozenset(("Pad",)))
def pad_vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    """Invert constant padding/cropping and sum the padded-value cotangent."""
    if node.attributes.get_string("mode", "constant") != "constant":
        raise NotImplementedError("VJP Pad requires mode='constant'")
    data = ctx.tape.op(
        "Pad",
        [gs[0], ctx.op("Neg", _input(node, 1)), ctx.scalar(_input(node), 0), *node.inputs[3:]],
        {"mode": "constant"},
    )
    contributions = {0: data} if 0 in indices else {}
    if 2 in indices:
        total = ctx.op("ReduceSum", gs[0], keepdims=0)
        inside = ctx.op("ReduceSum", data, keepdims=0)
        contributions[2] = ctx.op("Sub", total, inside)
    return contributions
