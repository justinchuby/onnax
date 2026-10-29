# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for layout operators."""

from __future__ import annotations

import onnx_ir as ir

from onnax._batching._context import _axis, _Context

OPS = frozenset(
    (
        "Concat",
        "ConstantOfShape",
        "Expand",
        "EyeLike",
        "Flatten",
        "Reshape",
        "Shape",
        "Size",
        "Split",
        "Squeeze",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
    )
)


def _reshape_or_expand(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data, shape = inputs
    length = ctx._shape_vector_length(shape)
    rank = length if node.op_type == "Reshape" else max(ctx.ranks[data], length)
    if ctx.mapped[data]:
        parts = [ctx._get_batch_shape()]
        if node.op_type == "Expand":
            if rank > length:
                parts.append(ctx._ints([1] * (rank - length)))
            node.replace_input_with(
                0, ctx._align(ctx.values[data], ctx.ranks[data], True, rank)
            )
        parts.append(ctx.values[shape])
        node.replace_input_with(1, ctx._emit("Concat", parts, {"axis": 0}))
    return rank


def _flatten(ctx: _Context, node: ir.Node, value: ir.Value) -> None:
    rank = ctx.ranks[value]
    axis = node.attributes.get_int("axis", 1)
    if not -rank <= axis <= rank:
        raise ir.passes.PreconditionError(
            f"Flatten axis {axis} is invalid for example rank {rank} "
            f"of {ctx._describe(value)}. Set axis in [{-rank}, {rank}]; "
            "it partitions example dimensions, not the inserted batch dimension."
        )
    if axis < 0:
        axis += rank
    if not ctx.mapped[value]:
        return
    sizes = [ctx._get_batch_shape()]
    for start, end in ((1, axis + 1), (axis + 1, rank + 1)):
        part = ctx._emit("Shape", [ctx.values[value]], {"start": start, "end": end})
        sizes.append(ctx._emit("ReduceProd", [part, ctx._ints([0])], {"keepdims": 1}))
    target = ctx._emit("Concat", sizes, {"axis": 0})
    node.op_type = "Reshape"
    node.attributes.clear()
    node.attributes["allowzero"] = ir.AttrInt64("allowzero", 1)
    node.resize_inputs(2)
    node.replace_input_with(1, target)


def _squeeze_or_unsqueeze(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    value = inputs[0]
    rank = ctx.ranks[value]
    if len(inputs) == 2:
        axes = ctx._constant_ints(inputs[1], f"{node.op_type} axes")
    else:
        shape = ctx._example_shape(value)
        if not shape.is_static():
            raise ir.passes.PreconditionError(
                f"Squeeze without axes requires a fully static example shape; "
                f"got {ctx._describe(value)}. Provide explicit shared constant INT64 axes "
                "for dimensions guaranteed to be singleton, or annotate a truly static shape."
            )
        axes = [i for i, dim in enumerate(shape) if dim == 1]
    output_rank = rank + len(axes) if node.op_type == "Unsqueeze" else rank - len(axes)
    axis_rank = output_rank if node.op_type == "Unsqueeze" else rank
    axes = [_axis(axis, axis_rank) for axis in axes]
    if len(set(axes)) != len(axes):
        raise ir.passes.PreconditionError(
            f"{node.op_type} axes must not contain duplicates after normalization; "
            f"got {axes!r} for example axis rank {axis_rank}. "
            "Supply unique axes in example coordinates, excluding the batch axis."
        )
    if node.outputs[0].shape is None:
        dims = list(ctx._example_shape(value))
        if node.op_type == "Unsqueeze":
            for axis in sorted(axes):
                dims.insert(axis, 1)
        else:
            dims = [dim for i, dim in enumerate(dims) if i not in axes]
        node.outputs[0].shape = ir.Shape(dims)
    if ctx.mapped[value]:
        node.resize_inputs(2)
        node.replace_input_with(1, ctx._ints([axis + 1 for axis in axes]))
    return output_rank


def _shape(ctx: _Context, node: ir.Node, value: ir.Value) -> None:
    rank = ctx.ranks[value]
    bounds = [node.attributes.get_int("start", 0), node.attributes.get_int("end", rank)]
    start, end = [min(max(bound + rank if bound < 0 else bound, 0), rank) for bound in bounds]
    if ctx.mapped[value]:
        # Materialize the example slice: runtime optimizers can incorrectly
        # fold Size(Shape(x, start=1)) to rank(x), including the batch axis.
        shape = ctx._emit("Shape", [ctx.values[value]])
        node.op_type = "Gather"
        node.attributes.clear()
        node.attributes["axis"] = ir.AttrInt64("axis", 0)
        node.resize_inputs(2)
        node.replace_input_with(0, shape)
        node.replace_input_with(1, ctx._ints(list(range(start + 1, end + 1))))
    node.outputs[0].dtype = ir.DataType.INT64
    if node.outputs[0].shape is None:
        node.outputs[0].shape = ir.Shape([max(end - start, 0)])


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op in ("Split", "TopK") and inputs:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]]
        axis = _axis(node.attributes.get_int("axis", -1 if op == "TopK" else 0), rank)
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
    elif op == "Tile" and len(inputs) == 2:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]]
        if mapped:
            target = ctx._emit("Concat", [ctx._ints([1]), ctx.values[inputs[1]]], {"axis": 0})
            node.replace_input_with(1, target)
    elif op == "ConstantOfShape" and len(inputs) == 1:
        rank = ctx._shape_vector_length(inputs[0])
        mapped = False
    elif op == "EyeLike" and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        if rank != 2:
            raise ir.passes.PreconditionError(
                f"EyeLike requires example rank 2; got {ctx._describe(inputs[0])}. "
                "Supply a matrix per example; rank-N EyeLike batching is unsupported."
            )
        if mapped:
            shape_value = ctx._emit("Shape", [ctx.values[inputs[0]]], {"start": 1})
            stub = ctx._emit("ConstantOfShape", [shape_value])
            stub = ctx._emit("CastLike", [stub, ctx.values[inputs[0]]])
            node.replace_input_with(0, stub)
        mapped = False
    elif op in ("Reshape", "Expand") and len(inputs) == 2:
        rank = _reshape_or_expand(ctx, node, inputs)
    elif op == "Flatten" and len(inputs) == 1:
        _flatten(ctx, node, inputs[0])
        rank = 2
    elif (op == "Squeeze" and 1 <= len(inputs) <= 2) or (
        op == "Unsqueeze" and len(inputs) == 2
    ):
        rank = _squeeze_or_unsqueeze(ctx, node, inputs)
    elif op == "Shape" and len(inputs) == 1:
        _shape(ctx, node, inputs[0])
        rank, mapped = 1, False
    elif op == "Size" and len(inputs) == 1:
        if mapped:
            shape = ctx._emit("Shape", [ctx.values[inputs[0]]], {"start": 1})
            node.op_type = "ReduceProd"
            node.attributes.clear()
            node.attributes["keepdims"] = ir.AttrInt64("keepdims", 0)
            node.resize_inputs(2)
            node.replace_input_with(0, shape)
            node.replace_input_with(1, ctx._ints([0]))
        rank, mapped = 0, False
    elif op == "Transpose" and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        perm = list(node.attributes.get_ints("perm", range(rank - 1, -1, -1)))
        if sorted(perm) != list(range(rank)):
            raise ir.passes.PreconditionError(
                f"Transpose perm must be a rank permutation of {list(range(rank))!r}; "
                f"got {perm!r} for {ctx._describe(inputs[0])}. "
                "Include every example axis exactly once; do not include the new batch axis."
            )
        if mapped:
            node.attributes["perm"] = ir.AttrInt64s("perm", [0, *(i + 1 for i in perm)])
    elif op == "Concat" and inputs:
        rank = ctx.ranks[inputs[0]]
        if any(ctx.ranks[value] != rank for value in inputs):
            raise ir.passes.PreconditionError(
                f"Concat inputs must have equal ranks; got "
                f"{[ctx._describe(v) for v in inputs]}. "
                "Reshape inputs to equal example ranks and matching non-concatenated "
                "dimensions before concatenation."
            )
        axis = _axis(node.attributes.get_int("axis"), rank)
        if node.outputs[0].shape is None:
            output_shape = ctx._example_shape(inputs[0])
            sizes = [ctx._example_shape(value)[axis] for value in inputs]
            output_shape[axis] = (
                sum(size for size in sizes if isinstance(size, int))
                if all(isinstance(size, int) for size in sizes)
                else None
            )
            node.outputs[0].shape = output_shape
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
            for i, value in enumerate(inputs):
                if not ctx.mapped[value]:
                    node.replace_input_with(i, ctx._repeat(ctx.values[value]))
    else:
        raise ctx._input_error(node)
    for output in node.outputs:
        ctx._register(output, rank, mapped)
