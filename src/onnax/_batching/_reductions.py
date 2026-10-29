# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for reductions operators."""

from __future__ import annotations

import onnx_ir as ir

from ._context import _axis, _Context
from ._registry import _REDUCTIONS

OPS = _REDUCTIONS | frozenset(
    ("ArgMax", "ArgMin", "CumSum", "Hardmax", "LogSoftmax", "Softmax")
)


def _reduce(ctx: _Context, node: ir.Node, inputs: list[ir.Value], mapped: bool) -> int:
    rank = ctx.ranks[inputs[0]]
    if len(inputs) == 2 and ctx._constant_tensor(inputs[1]) is None:
        axes_value = inputs[1]
        if (
            ctx.mapped[axes_value]
            or ctx.ranks[axes_value] != 1
            or axes_value.dtype != ir.DataType.INT64
        ):
            raise ir.passes.PreconditionError(
                f"Reduction axes must be a shared 1-D INT64 tensor; "
                f"got {ctx._describe(axes_value)}. Supply axes with dtype=INT64, shape=[N], "
                "and in_axes=None if shared. Per-example axes require independent execution."
            )
        keepdims = node.attributes.get_int("keepdims", 1)
        noop = node.attributes.get_int("noop_with_empty_axes", 0)
        length = ctx._example_shape(axes_value)[0]
        if keepdims:
            output_rank = rank
        elif isinstance(length, int):
            output_rank = rank - length if length or noop else 0
        else:
            raise ir.passes.PreconditionError(
                "Runtime reduction axes with keepdims=0 require a known axes length; "
                f"got {ctx._describe(axes_value)}. Annotate shape=[N] with a fixed N "
                "matching the runtime axes count. Use keepdims=1 only if retaining "
                "reduced dimensions matches the intended output interface."
            )
        if not mapped:
            return output_rank
        # Normalize negatives in example coordinates, then skip the batch axis.
        runtime_axes = ctx.values[axes_value]
        normalized = ctx._emit(
            "Where",
            [
                ctx._emit("Less", [runtime_axes, ctx._scalar_int(0)]),
                ctx._emit("Add", [runtime_axes, ctx._scalar_int(rank)]),
                runtime_axes,
            ],
        )
        shifted = ctx._emit("Add", [normalized, ctx._scalar_int(1)])
        if not noop:
            # Empty axes mean all *example* dimensions, never the batch.
            # A nonempty sentinel also implements scalar reductions.
            if rank == 0:
                node.replace_input_with(0, ctx._unsqueeze(ctx.values[inputs[0]], [1]))
                shifted = ctx._ints([1])
                node.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
            else:
                size = ctx._emit("Size", [runtime_axes])
                stop = ctx._emit(
                    "Where",
                    [
                        ctx._emit("Equal", [size, ctx._scalar_int(0)]),
                        ctx._scalar_int(rank + 1),
                        ctx._scalar_int(1),
                    ],
                )
                all_axes = ctx._emit("Range", [ctx._scalar_int(1), stop, ctx._scalar_int(1)])
                shifted = ctx._emit("Concat", [shifted, all_axes], {"axis": 0})
        node.replace_input_with(1, shifted)
        return output_rank
    axes = ctx._constant_ints(inputs[1], "Reduction axes") if len(inputs) == 2 else []
    noop = node.attributes.get_int("noop_with_empty_axes", 0)
    if not axes and noop:
        return rank
    axes = [_axis(axis, rank) for axis in axes] if axes else list(range(rank))
    if len(set(axes)) != len(axes):
        raise ir.passes.PreconditionError(
            f"Reduction axes must not contain duplicates after normalization; got "
            f"{axes!r} for example rank {rank}. Supply each reduced example axis once."
        )
    if mapped:
        if not axes:
            # A singleton example dimension preserves scalar ReduceL1,
            # ReduceLogSum, etc., without reducing the batch dimension.
            node.replace_input_with(0, ctx._unsqueeze(ctx.values[inputs[0]], [1]))
            shifted = ctx._ints([1])
            node.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
        else:
            shifted = ctx._ints([axis + 1 for axis in axes])
        node.resize_inputs(2)
        node.replace_input_with(1, shifted)
    keepdims = node.attributes.get_int("keepdims", 1)
    return rank if keepdims else rank - len(axes)


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op == "CumSum" and len(inputs) == 2:
        rank = ctx.ranks[inputs[0]]
        axis = ctx._constant_axis(inputs[1], rank)
        if mapped:
            node.replace_input_with(1, ctx._scalar_int(axis + 1))
    elif op in ("ArgMax", "ArgMin") and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        axis = _axis(node.attributes.get_int("axis", 0), rank)
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        keepdims = node.attributes.get_int("keepdims", 1)
        rank -= int(not keepdims)
    elif op in ("Softmax", "LogSoftmax", "Hardmax") and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        axis = _axis(node.attributes.get_int("axis", -1), rank)
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
    elif op in _REDUCTIONS and 1 <= len(inputs) <= 2:
        rank = _reduce(ctx, node, inputs, mapped)
    else:
        raise ctx._input_error(node)
    for output in node.outputs:
        ctx._register(output, rank, mapped)
