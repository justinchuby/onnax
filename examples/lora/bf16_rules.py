# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Explicit example-only BF16 MatMul semantics with FP32 straight-through derivatives."""

from __future__ import annotations

import onnx_ir as ir

import onnax

DOMAIN = "onnax.example.precision"
OP = "MatMulBF16"


def rounded(ctx, value):
    """Expose a BF16-rounded primal operand as FP32 for derivative arithmetic."""
    return ctx.op("Cast", ctx.op("Cast", value, to=ir.DataType.BFLOAT16), to=ir.DataType.FLOAT)


def operands(ctx, node):
    values = [ctx.input(node, i) for i in (0, 1)]
    if any(
        value.dtype != ir.DataType.FLOAT or value.shape is None or len(value.shape) != 2
        for value in values
    ):
        raise TypeError("Example MatMulBF16 rules require rank-two FP32 boundary operands.")
    return values


@onnax.register_jvp(DOMAIN, OP, since_version=1, until_version=1)
def matmul_jvp(ctx, node, tangents):
    a, b = operands(ctx, node)
    terms = []
    if 0 in tangents:
        terms.append(ctx.op("MatMul", tangents[0], rounded(ctx, b)))
    if 1 in tangents:
        terms.append(ctx.op("MatMul", rounded(ctx, a), tangents[1]))
    return [ctx.sum(terms)]


@onnax.register_vjp(DOMAIN, OP, since_version=1, until_version=1)
def matmul_vjp(ctx, node, cotangents, input_indices):
    a, b = operands(ctx, node)
    g = cotangents[0]
    contributions = {}
    if 0 in input_indices:
        contributions[0] = ctx.op("MatMul", g, ctx.transpose(rounded(ctx, b), 2))
    if 1 in input_indices:
        contributions[1] = ctx.op("MatMul", ctx.transpose(rounded(ctx, a), 2), g)
    return contributions


def lower_matmuls(model):
    """Replace this example's FP32 matrix calls without changing its public boundary.

    The function body computes actual BF16-input/BF16-output ONNX MatMul.
    Registered derivatives use identity derivatives for the rounding boundaries.
    They are a declared straight-through surrogate, not rounding derivatives.
    """
    result = model.clone()
    if any(node.domain or node.op_type in ("Gemm",) for node in result.graph):
        raise ValueError("Expected the example's standard-domain MatMul graph.")
    if any(value.dtype != ir.DataType.FLOAT for value in result.graph.inputs) or any(
        value.dtype != ir.DataType.FLOAT
        and value.dtype != ir.DataType.BOOL
        and (value.dtype is None or not value.dtype.is_integer())
        for value in result.graph.initializers.values()
    ):
        raise TypeError("Expected FP32 model inputs and floating initializers.")
    if any(
        value is None or value.dtype not in (None, ir.DataType.FLOAT)
        for node in result.graph
        if node.op_type == "MatMul"
        for value in node.inputs
    ):
        raise TypeError("Expected FP32 matrix operands.")
    args = [ir.val(n) for n in ("a", "b")]
    tape = ir.tape.Tape()
    reduced = [tape.op("Cast", [arg], {"to": int(ir.DataType.BFLOAT16)}) for arg in args]
    product = tape.op("MatMul", reduced)
    output = tape.op("Cast", [product], {"to": int(ir.DataType.FLOAT)})
    function = ir.Function(
        DOMAIN,
        OP,
        graph=ir.Graph(
            args,
            [output],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="bf16_matrix_product",
        ),
        attributes=[],
    )
    result.functions[function.identifier()] = function
    result.graph.opset_imports[DOMAIN] = 1
    for node in result.graph:
        if node.op_type == "MatMul":
            node.domain = DOMAIN
            node.op_type = OP
    return result
