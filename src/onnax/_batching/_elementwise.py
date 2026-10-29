# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for elementwise operators."""

from __future__ import annotations

import onnx_ir as ir

from onnax._batching._context import _Context
from onnax._batching._registry import _BINARY, _UNARY, _VARIADIC

OPS = (
    _UNARY
    | _BINARY
    | _VARIADIC
    | frozenset(("CastLike", "Clip", "Constant", "PRelu", "Where"))
)


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op == "Constant":
        attr = node.attributes.get("value")
        if attr is None:
            for key in ("value_int", "value_ints", "value_float", "value_floats"):
                literal = node.attributes.get(key)
                if literal is None:
                    continue
                if key == "value_int":
                    tensor = ir.tensor(literal.as_int(), dtype=ir.DataType.INT64)
                elif key == "value_ints":
                    tensor = ir.tensor(literal.as_ints(), dtype=ir.DataType.INT64)
                elif key == "value_float":
                    tensor = ir.tensor(literal.as_float(), dtype=ir.DataType.FLOAT)
                else:
                    tensor = ir.tensor(literal.as_floats(), dtype=ir.DataType.FLOAT)
                node.attributes.clear()
                attr = ir.AttrTensor("value", tensor)
                node.attributes["value"] = attr
                break
        if inputs or attr is None or attr.type != ir.AttributeType.TENSOR:
            raise ir.passes.PreconditionError("Only tensor-valued Constant is supported.")
        tensor = attr.as_tensor()
        rank = len(tensor.shape)
        node.outputs[0].type = ir.TensorType(tensor.dtype)
        if node.outputs[0].shape is None:
            node.outputs[0].shape = ir.Shape(
                d if isinstance(d, int) else d.value for d in tensor.shape.dims
            )
    elif op in _UNARY and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]]
        if node.outputs[0].shape is None:
            node.outputs[0].shape = ctx._example_shape(inputs[0])
    elif op == "CastLike" and len(inputs) == 2:
        rank = ctx.ranks[inputs[0]]
        mapped = ctx.mapped[inputs[0]]
        if node.outputs[0].shape is None:
            node.outputs[0].shape = ctx._example_shape(inputs[0])
    elif op == "Clip" and 1 <= len(inputs) <= 3:
        if any(ctx.mapped[value] or ctx.ranks[value] != 0 for value in inputs[1:]):
            raise ir.passes.PreconditionError("Clip bounds must be shared scalars.")
        rank = ctx.ranks[inputs[0]]
    elif op == "PRelu" and len(inputs) == 2:
        data, slope = inputs
        rank = ctx.ranks[data]
        if ctx.ranks[slope] > rank:
            raise ir.passes.PreconditionError("PRelu slope rank must not exceed data rank.")
        if mapped:
            if not ctx.mapped[data]:
                node.replace_input_with(0, ctx._repeat(ctx.values[data]))
            node.replace_input_with(
                1,
                ctx._align(ctx.values[slope], ctx.ranks[slope], ctx.mapped[slope], rank),
            )
    elif (
        (op in _BINARY and len(inputs) == 2)
        or (op == "Where" and len(inputs) == 3)
        or (op in _VARIADIC and inputs)
    ):
        rank = max(ctx.ranks[value] for value in inputs)
        if mapped:
            for i, value in enumerate(inputs):
                node.replace_input_with(
                    i,
                    ctx._align(ctx.values[value], ctx.ranks[value], ctx.mapped[value], rank),
                )
    else:
        raise ir.passes.PreconditionError(
            f"VmapPass has no batching rule for {op!r} with {len(inputs)} inputs."
        )
    for output in node.outputs:
        ctx._register(output, rank, mapped)
