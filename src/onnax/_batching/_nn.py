# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for nn operators."""

from __future__ import annotations

import onnx_ir as ir

from ._context import _axis, _Context
from ._registry import _NATIVE_BATCH

OPS = _NATIVE_BATCH | frozenset(
    (
        "DequantizeLinear",
        "Dropout",
        "LayerNormalization",
        "LpNormalization",
        "MeanVarianceNormalization",
        "QuantizeLinear",
        "Resize",
        "Trilu",
    )
)


def _native_batch(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    data = inputs[0]
    rank = ctx.ranks[data]
    if rank < 3:
        raise ir.passes.PreconditionError(f"{node.op_type} requires example rank >= 3.")
    if node.op_type in ("DepthToSpace", "SpaceToDepth") and rank != 4:
        raise ir.passes.PreconditionError(f"{node.op_type} requires example rank 4.")
    native_positions = (0, 1) if node.op_type == "GridSample" else (0,)
    ctx._require_shared(
        [value for i, value in enumerate(inputs) if i not in native_positions],
        node.op_type,
    )
    training_mode = node.attributes.get_int("training_mode", 0)
    if node.op_type == "BatchNormalization" and training_mode:
        raise ir.passes.PreconditionError("Mapped BatchNormalization requires inference mode.")
    if not any(ctx.mapped[value] for value in inputs):
        return rank
    native_size = None
    for i in native_positions:
        value = inputs[i]
        batched = ctx.values[value] if ctx.mapped[value] else ctx._repeat(ctx.values[value])
        shape = ctx._emit("Shape", [batched])
        size = ctx._emit("Gather", [shape, ctx._ints([1])], {"axis": 0})
        if native_size is None:
            native_size = size
        merged = ctx._emit("Mul", [ctx._get_batch_shape(), size])
        tail = ctx._emit("Shape", [batched], {"start": 2})
        target = ctx._emit("Concat", [merged, tail], {"axis": 0})
        node.replace_input_with(i, ctx._emit("Reshape", [batched, target], {"allowzero": 1}))
    result = ctx._emit(node.op_type, list(node.inputs), node.attributes)
    tail = ctx._emit("Shape", [result], {"start": 1})
    assert native_size is not None
    target = ctx._emit("Concat", [ctx._get_batch_shape(), native_size, tail], {"axis": 0})
    node.op_type = "Reshape"
    node.attributes.clear()
    node.attributes["allowzero"] = ir.AttrInt64("allowzero", 1)
    node.resize_inputs(2)
    node.replace_input_with(0, result)
    node.replace_input_with(1, target)
    return rank


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op in ("QuantizeLinear", "DequantizeLinear") and 2 <= len(inputs) <= 3:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]]
        block_size = node.attributes.get_int("block_size", 0)
        if ctx.ranks[inputs[1]] > 1 or block_size:
            raise ir.passes.PreconditionError(
                f"{op} supports shared scalar or per-axis parameters, not blocked quantization."
            )
        if mapped and ctx.ranks[inputs[1]] == 1:
            axis = _axis(node.attributes.get_int("axis", 1), rank)
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
    elif op in _NATIVE_BATCH and inputs:
        rank = _native_batch(ctx, node, inputs)
    elif op in ("Resize", "Trilu", "LpNormalization") and inputs:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]]
        if op == "Resize":
            axes = node.attributes.get_ints("axes", range(rank))
            axes = [_axis(axis, rank) for axis in axes]
            if mapped:
                node.attributes["axes"] = ir.AttrInt64s("axes", [axis + 1 for axis in axes])
        elif op == "LpNormalization":
            axis = _axis(node.attributes.get_int("axis", -1), rank)
            if mapped:
                node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
        elif rank < 2:
            raise ir.passes.PreconditionError("Trilu requires example rank >= 2.")
    elif op == "LayerNormalization" and inputs:
        ctx._require_shared(inputs[1:], op)
        rank = ctx.ranks[inputs[0]]
        axis = _axis(node.attributes.get_int("axis", -1), rank)
        if mapped:
            node.attributes["axis"] = ir.AttrInt64("axis", axis + 1)
    elif op == "MeanVarianceNormalization" and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        axes = [_axis(axis, rank) for axis in node.attributes.get_ints("axes", (0, 2, 3))]
        if mapped:
            node.attributes["axes"] = ir.AttrInt64s("axes", [axis + 1 for axis in axes])
    elif op == "Dropout" and inputs:
        training = node.inputs[2] if len(node.inputs) > 2 else None
        if training is not None:
            training_tensor = ctx._constant_tensor(training)
            if (
                training_tensor is None
                or training_tensor.dtype != ir.DataType.BOOL
                or len(training_tensor.shape) != 0
                or bool(training_tensor.numpy().item())
            ):
                raise ir.passes.PreconditionError(
                    "Dropout requires a constant false or omitted training_mode."
                )
        rank = ctx.ranks[inputs[0]]
        mapped = ctx.mapped[inputs[0]]
        node.resize_inputs(1)
    else:
        raise ir.passes.PreconditionError(
            f"VmapPass has no batching rule for {op!r} with {len(inputs)} inputs."
        )
    for output in node.outputs:
        ctx._register(output, rank, mapped)
