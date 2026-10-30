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
        if node.attributes.get("reduction") is None or (
            node.attributes["reduction"].as_string() != "add"
        ):
            reduction = node.attributes.get("reduction")
            raise NotImplementedError(
                "JVP ScatterND requires reduction='add'; "
                f"got reduction={reduction.as_string() if reduction is not None else 'none (default)'!r} "
                f"(node {node.name!r}). Only additive ScatterND differentiation is supported. "
                "Keep the original reduction and use a separate derivative implementation "
                "for overwrite/mul/min/max semantics; changing to 'add' changes the function."
            )
        return [
            ctx.op(
                "ScatterND",
                d[0] if 0 in d else ctx.like(_input(node)),
                _input(node, 1),
                d[2] if 2 in d else ctx.like(_input(node, 2)),
                reduction="add",
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
    elif op in ("Cast",):
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
