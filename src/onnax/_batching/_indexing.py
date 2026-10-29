# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for indexing operators."""

from __future__ import annotations

import onnx_ir as ir

from ._context import _axis, _Context

OPS = frozenset(
    (
        "Gather",
        "GatherElements",
        "GatherND",
        "OneHot",
        "Pad",
        "ScatterElements",
        "ScatterND",
        "Slice",
    )
)


def _index_aligned(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    rank = ctx.ranks[inputs[0]]
    if any(ctx.ranks[value] != rank for value in inputs):
        raise ir.passes.PreconditionError(f"{node.op_type} operands must have equal ranks.")
    axis = _axis(node.attributes.get_int("axis", 0), rank)
    if any(ctx.mapped[value] for value in inputs):
        node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        for i, value in enumerate(inputs):
            if not ctx.mapped[value]:
                node.replace_input_with(i, ctx._repeat(ctx.values[value]))
    return rank


def _gather_nd(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data, indices = inputs
    batch_dims = node.attributes.get_int("batch_dims", 0)
    index_shape = ctx._example_shape(indices)
    if not len(index_shape):
        raise ir.passes.PreconditionError("GatherND indices must have rank >= 1.")
    depth = index_shape[-1]
    if isinstance(depth, int):
        rank = ctx.ranks[data] + ctx.ranks[indices] - depth - batch_dims - 1
    elif node.outputs[0].shape is not None:
        rank = len(node.outputs[0].shape)
    else:
        raise ir.passes.PreconditionError("GatherND needs a known index depth or output rank.")
    if any(ctx.mapped[value] for value in inputs):
        node.attributes["batch_dims"] = ir.AttrInt64("batch_dims", batch_dims + 1)
        for i, value in enumerate(inputs):
            if not ctx.mapped[value]:
                node.replace_input_with(i, ctx._repeat(ctx.values[value]))
    return rank


def _scatter_nd(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    rank = ctx.ranks[inputs[0]]
    if not any(ctx.mapped[value] for value in inputs):
        return rank
    values = [
        ctx.values[value] if ctx.mapped[value] else ctx._repeat(ctx.values[value])
        for value in inputs
    ]
    indices = values[1]
    index_rank = ctx.ranks[inputs[1]]
    if index_rank < 1:
        raise ir.passes.PreconditionError("ScatterND indices must have rank >= 1.")
    batch = ctx._emit("Squeeze", [ctx._get_batch_shape(), ctx._ints([0])])
    ids = ctx._emit("Range", [ctx._scalar_int(0), batch, ctx._scalar_int(1)])
    id_shape = ctx._emit(
        "Concat", [ctx._get_batch_shape(), ctx._ints([1] * index_rank)], {"axis": 0}
    )
    ids = ctx._emit("Reshape", [ids, id_shape], {"allowzero": 1})
    prefix = ctx._emit("Shape", [indices], {"end": index_rank})
    target = ctx._emit("Concat", [prefix, ctx._ints([1])], {"axis": 0})
    ids = ctx._emit("Expand", [ids, target])
    values[1] = ctx._emit("Concat", [ids, indices], {"axis": -1})
    for i, value in enumerate(values):
        node.replace_input_with(i, value)
    return rank


def _slice(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data = inputs[0]
    rank = ctx.ranks[data]
    ctx._require_shared(inputs[1:], "Slice")
    if not ctx.mapped[data]:
        return rank
    axes_value = node.inputs[3] if len(node.inputs) > 3 else None
    if axes_value is not None:
        axes = ctx._constant_ints(axes_value, "Slice axes")
    else:
        starts_shape = ctx.values[inputs[1]].shape
        if (
            starts_shape is None
            or len(starts_shape) != 1
            or not isinstance(starts_shape[0], int)
        ):
            raise ir.passes.PreconditionError(
                "Slice without axes needs a known starts length."
            )
        axes = list(range(starts_shape[0]))
    axes = [_axis(axis, rank) + 1 for axis in axes]
    if len(set(axes)) != len(axes):
        raise ir.passes.PreconditionError("Slice axes must not contain duplicates.")
    for i in (1, 2, 4):
        if i < len(node.inputs):
            control = node.inputs[i]
            if control is not None:
                node.replace_input_with(i, ctx._cast_int64(control))
    if len(node.inputs) < 4:
        node.resize_inputs(4)
    node.replace_input_with(3, ctx._ints(axes))
    return rank


def _pad(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data = inputs[0]
    rank = ctx.ranks[data]
    ctx._require_shared(inputs[1:], "Pad")
    if not ctx.mapped[data]:
        return rank
    axes = node.inputs[3] if len(node.inputs) > 3 else None
    if axes is not None:
        shifted = [_axis(axis, rank) + 1 for axis in ctx._constant_ints(axes, "Pad axes")]
        node.replace_input_with(3, ctx._ints(shifted))
    else:
        pads = node.inputs[1]
        assert pads is not None
        before = ctx._emit("Slice", [pads, ctx._ints([0]), ctx._ints([rank])])
        after = ctx._emit("Slice", [pads, ctx._ints([rank]), ctx._ints([2 * rank])])
        target = ctx._emit(
            "Concat", [ctx._ints([0]), before, ctx._ints([0]), after], {"axis": 0}
        )
        node.replace_input_with(1, target)
    return rank


def _gather(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data, indices = inputs
    data_rank, index_rank = ctx.ranks[data], ctx.ranks[indices]
    axis = _axis(node.attributes.get_int("axis", 0), data_rank)
    rank = data_rank + index_rank - 1
    if node.outputs[0].shape is None:
        dims = list(ctx._example_shape(data))
        node.outputs[0].shape = ir.Shape(
            [*dims[:axis], *ctx._example_shape(indices), *dims[axis + 1 :]]
        )
    if not ctx.mapped[indices]:
        if ctx.mapped[data]:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        return rank
    if ctx.mapped[data]:
        data_perm = [0, axis + 1, *range(1, axis + 1), *range(axis + 2, data_rank + 1)]
        data_value = ctx._transpose(ctx.values[data], data_perm)
        indices_value = ctx._emit(
            "Cast", [ctx.values[indices]], {"to": int(ir.DataType.INT64)}
        )
        indices_value = ctx._unsqueeze(indices_value, [index_rank + 1])
        gathered = ctx._emit("GatherND", [data_value, indices_value], {"batch_dims": 1})
        perm = [
            0,
            *range(index_rank + 1, index_rank + axis + 1),
            *range(1, index_rank + 1),
            *range(index_rank + axis + 1, rank + 1),
        ]
    else:
        gathered = ctx._emit("Gather", [ctx.values[data], ctx.values[indices]], {"axis": axis})
        perm = [axis, *range(axis), *range(axis + 1, rank + 1)]
    node.op_type = "Transpose"
    node.attributes.clear()
    node.attributes["perm"] = ir.AttrInt64s("perm", perm)
    node.resize_inputs(1)
    node.replace_input_with(0, gathered)
    return rank


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op == "Slice" and 3 <= len(inputs) <= 5:
        rank = _slice(ctx, node, inputs)
    elif op == "Pad" and 2 <= len(inputs) <= 4:
        rank = _pad(ctx, node, inputs)
    elif op == "OneHot" and len(inputs) == 3:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]] + 1
        axis = _axis(node.attributes.get_int("axis", -1), rank)
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
    elif op == "GatherElements" and len(inputs) == 2:
        rank = _index_aligned(ctx, node, inputs)
    elif op == "ScatterElements" and len(inputs) == 3:
        rank = _index_aligned(ctx, node, inputs)
    elif op == "GatherND" and len(inputs) == 2:
        rank = _gather_nd(ctx, node, inputs)
    elif op == "ScatterND" and len(inputs) == 3:
        rank = _scatter_nd(ctx, node, inputs)
    elif op == "Gather" and len(inputs) == 2:
        rank = _gather(ctx, node, inputs)
    else:
        raise ir.passes.PreconditionError(
            f"VmapPass has no batching rule for {op!r} with {len(inputs)} inputs."
        )
    for output in node.outputs:
        ctx._register(output, rank, mapped)
