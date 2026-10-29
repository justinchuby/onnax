# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batching rules for linalg operators."""

from __future__ import annotations

import collections
import string

import onnx_ir as ir

from onnax._batching._context import _Context

OPS = frozenset(("Det", "Einsum", "Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"))


def _einsum(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> int:
    equation = node.attributes.get_string("equation").replace(" ", "")
    left, separator, right = equation.partition("->")
    terms = left.split(",")
    if len(terms) != len(inputs):
        raise ir.passes.PreconditionError(
            f"Einsum equation {equation!r} must match the input count: "
            f"got {len(terms)} terms for {len(inputs)} inputs. "
            "Supply one comma-separated term per input and a valid output term."
        )
    if not separator:
        counts = collections.Counter(char for char in left if char in string.ascii_letters)
        right = ("..." if "..." in left else "") + "".join(
            sorted(char for char, count in counts.items() if count == 1)
        )
    ellipsis = max(
        (
            ctx.ranks[value] - len(term.replace("...", ""))
            for value, term in zip(inputs, terms)
            if "..." in term
        ),
        default=0,
    )
    rank = len(right.replace("...", "")) + (ellipsis if "..." in right else 0)
    if any(ctx.mapped[value] for value in inputs):
        label = next((char for char in string.ascii_letters if char not in equation), None)
        if label is None:
            raise ir.passes.PreconditionError(
                f"Einsum needs an unused label for the batch axis; equation={equation!r} "
                "uses all 52 ASCII labels. Rewrite the equation with fewer distinct labels "
                "if equivalent, or execute examples independently."
            )
        terms = [
            label + term if ctx.mapped[value] else term for value, term in zip(inputs, terms)
        ]
        node.attributes["equation"] = ir.AttrString(
            "equation", ",".join(terms) + "->" + label + right
        )
    return rank


def _gemm(ctx: _Context, node: ir.Node, inputs: list[ir.Value]) -> None:
    if any(ctx.ranks[value] != 2 for value in inputs[:2]):
        raise ir.passes.PreconditionError(
            f"Gemm matrix operands must have example rank 2; got "
            f"{[ctx._describe(v) for v in inputs[:2]]}. "
            "Supply two matrices per example, or use MatMul for vector/batched-matrix semantics."
        )
    left, right = [ctx.values[value] for value in inputs[:2]]
    for i, key in enumerate(("transA", "transB")):
        if node.attributes.get_int(key, 0):
            perm = [0, 2, 1] if ctx.mapped[inputs[i]] else [1, 0]
            if i == 0:
                left = ctx._transpose(left, perm)
            else:
                right = ctx._transpose(right, perm)
    product = ctx._emit("MatMul", [left, right])
    for key, value in (
        ("alpha", product),
        ("beta", ctx.values[inputs[2]] if len(inputs) == 3 else None),
    ):
        coefficient = node.attributes.get_float(key, 1.0)
        if value is not None and coefficient != 1:
            scalar = ctx._emit("Constant", [], {"value": ir.tensor(coefficient)})
            scalar = ctx._emit("CastLike", [scalar, left])
            scaled = ctx._emit("Mul", [value, scalar])
            if key == "alpha":
                product = scaled
            else:
                node.replace_input_with(2, scaled)
    if len(inputs) == 3:
        bias = node.inputs[2]
        assert bias is not None
        bias = ctx._align(bias, ctx.ranks[inputs[2]], ctx.mapped[inputs[2]], 2)
        node.op_type = "Add"
        node.resize_inputs(2)
        node.replace_input_with(1, bias)
    else:
        node.op_type = "Identity"
        node.resize_inputs(1)
    node.attributes.clear()
    node.replace_input_with(0, product)


def _matmul(
    ctx: _Context,
    node: ir.Node,
    inputs: list[ir.Value],
    mapped: bool,
    positions: tuple[int, int] = (0, 1),
) -> int:
    left, right = inputs
    left_rank, right_rank = ctx.ranks[left], ctx.ranks[right]
    if min(left_rank, right_rank) < 1:
        raise ir.passes.PreconditionError(
            f"{node.op_type} operands must have rank >= 1; got "
            f"{[ctx._describe(v) for v in inputs]}. "
            "Supply vectors/matrices; use Mul for scalar multiplication."
        )
    target = max(left_rank, right_rank, 2)
    rank = target - int(left_rank == 1) - int(right_rank == 1)
    if not mapped:
        return rank
    left_value, right_value = ctx.values[left], ctx.values[right]
    if left_rank == 1:
        left_value = ctx._unsqueeze(left_value, [int(ctx.mapped[left])])
    if right_rank == 1:
        right_value = ctx._unsqueeze(right_value, [right_rank + int(ctx.mapped[right])])
    node.replace_input_with(
        positions[0], ctx._align(left_value, max(left_rank, 2), ctx.mapped[left], target)
    )
    node.replace_input_with(
        positions[1],
        ctx._align(right_value, max(right_rank, 2), ctx.mapped[right], target),
    )
    if left_rank == 1 or right_rank == 1:
        squeeze_axes = []
        if left_rank == 1:
            squeeze_axes.append(target - 1)
        if right_rank == 1:
            squeeze_axes.append(target)
        product = ctx._emit(node.op_type, list(node.inputs), node.attributes)
        product.type = node.outputs[0].type
        product.shape = ir.Shape([ctx.axis_size, *([None] * target)])
        axes = ctx._ints(squeeze_axes)
        node.op_type = "Squeeze"
        node.attributes.clear()
        node.resize_inputs(2)
        node.replace_input_with(0, product)
        node.replace_input_with(1, axes)
    return rank


def apply(ctx: _Context, node: ir.Node) -> None:
    inputs, mapped = ctx._prepare_node(node)
    op = node.op_type
    if op == "MatMul" and len(inputs) == 2:
        rank = _matmul(ctx, node, inputs, mapped)
    elif op == "MatMulInteger" and 2 <= len(inputs) <= 4:
        ctx._require_shared(inputs[2:], op)
        rank = _matmul(ctx, node, inputs[:2], mapped)
    elif op == "QLinearMatMul" and len(inputs) == 8:
        ctx._require_shared([value for i, value in enumerate(inputs) if i not in (0, 3)], op)
        rank = _matmul(ctx, node, [inputs[0], inputs[3]], mapped, positions=(0, 3))
    elif op == "Gemm" and 2 <= len(inputs) <= 3:
        _gemm(ctx, node, inputs)
        rank = 2
    elif op == "Einsum" and inputs:
        rank = _einsum(ctx, node, inputs)
    elif op == "Det" and len(inputs) == 1:
        rank = ctx.ranks[inputs[0]] - 2
        if rank < 0:
            raise ir.passes.PreconditionError(
                f"Det requires example rank >= 2; got {ctx._describe(inputs[0])}. "
                "Supply square matrices in the last two example dimensions."
            )
    else:
        raise ctx._input_error(node)
    for output in node.outputs:
        ctx._register(output, rank, mapped)
