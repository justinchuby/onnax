# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable ONNX losses that preserve small gradients in FP32.

Log1p/Expm1 use compensated standard-ONNX function bodies. Hard-label CE
and distillation use analytic global JVP/VJP rules. Preserve these functions during optimization before further AD.
Teacher logits are held fixed in the distillation loss.
"""

import numpy as np
import onnx_ir as ir

import onnax

DOMAIN = "onnax.example.stable_loss"


def _name(op, dtype):
    return op + ("F32" if dtype == ir.DataType.FLOAT else "F64")


def _call(ctx, op, x):
    output = ctx.op(_name(op, x.dtype), x, domain=DOMAIN)
    output.type, output.shape = ir.TensorType(x.dtype), x.shape
    return output


def _mask(ctx, x, indices, axis=-1):
    ones = ctx.op(
        "ConstantOfShape", ctx.op("Shape", indices), value=ir.tensor([1], dtype=x.dtype)
    )
    return ctx.op(
        "Cast",
        ctx.op("ScatterElements", ctx.like(x), indices, ones, axis=axis),
        to=ir.DataType.BOOL,
    )


def log_softmax(ctx, x):
    axis, zero = ctx.constant([-1]), ctx.scalar(x, 0)
    pivot = ctx.op("ArgMax", x, axis=-1, keepdims=1)
    mask = _mask(ctx, x, pivot)
    maximum = ctx.op("ReduceSum", ctx.op("Where", mask, x, zero), axis, keepdims=1)
    shifted = ctx.op("Where", mask, zero, ctx.op("Sub", x, maximum))
    mass = ctx.op(
        "ReduceSum", ctx.op("Where", mask, zero, ctx.op("Exp", shifted)), axis, keepdims=1
    )
    mass.type = ir.TensorType(x.dtype)
    correction = _call(ctx, "Log1p", mass)
    result = ctx.op("Sub", shifted, correction)
    result.type, result.shape = ir.TensorType(x.dtype), x.shape
    return result


def _hard_terms(ctx, logits, labels):
    zero, one = ctx.scalar(logits, 0), ctx.scalar(logits, 1)
    axes = ctx.constant([-1])
    valid = ctx.op("GreaterOrEqual", labels, ctx.constant(0))
    # Ignored rows must be finite before exponentials and their pullbacks.
    dtype, shape = logits.dtype, logits.shape
    logits = ctx.op("Where", ctx.op("Unsqueeze", valid, axes), logits, zero)
    logits.type, logits.shape = ir.TensorType(dtype), shape
    safe = ctx.op("Max", labels, ctx.constant(0))
    index = ctx.op("Unsqueeze", safe, axes)
    target = _mask(ctx, logits, index)
    target_value = ctx.op("ReduceSum", ctx.op("Where", target, logits, zero), axes, keepdims=1)
    other_logits = ctx.op("Where", target, ctx.scalar(logits, -np.inf), logits)
    other = _mask(ctx, logits, ctx.op("ArgMax", other_logits, axis=-1, keepdims=1))
    maximum = ctx.op("ReduceSum", ctx.op("Where", other, logits, zero), axes, keepdims=1)
    shifted = ctx.op("Where", target, zero, ctx.op("Sub", logits, maximum))
    mass = ctx.op(
        "ReduceSum", ctx.op("Where", target, zero, ctx.op("Exp", shifted)), axes, keepdims=1
    )
    depth = ctx.op("Gather", ctx.op("Shape", logits), ctx.constant(-1), axis=0)
    has_others = ctx.op("Greater", depth, ctx.constant(1))
    safe_mass = ctx.op("Where", has_others, mass, one)
    margin = ctx.op("Add", ctx.op("Sub", maximum, target_value), ctx.op("Log", safe_mass))
    return target, valid, has_others, shifted, safe_mass, margin


def _hard_body(ctx, logits, labels):
    _, valid, has_others, _, _, margin = _hard_terms(ctx, logits, labels)
    zero = ctx.scalar(logits, 0)
    loss = ctx.op("Where", has_others, ctx.op("Softplus", margin), zero)
    return ctx.op("Where", valid, ctx.op("Squeeze", loss, ctx.constant([-1])), zero)


def _hard_gradient(ctx, logits, labels):
    target, valid, has_others, shifted, mass, margin = _hard_terms(ctx, logits, labels)
    zero, one = ctx.scalar(logits, 0), ctx.scalar(logits, 1)
    # Stable sigmoid retains negative-margin tails on CPU and MLX.
    positive = ctx.op("GreaterOrEqual", margin, zero)
    magnitude = ctx.op("Where", positive, margin, ctx.op("Neg", margin))
    tail = ctx.op("Exp", ctx.op("Neg", magnitude))
    denominator = ctx.op("Add", one, tail)
    probability = ctx.op(
        "Where",
        positive,
        ctx.op("Div", one, denominator),
        ctx.op("Div", tail, denominator),
    )
    probability = ctx.op("Where", has_others, probability, zero)
    other = ctx.op("Div", ctx.op("Exp", shifted), mass)
    derivative = ctx.op(
        "Where", target, ctx.op("Neg", probability), ctx.op("Mul", other, probability)
    )
    return ctx.op("Where", ctx.op("Unsqueeze", valid, ctx.constant([-1])), derivative, zero)


def hard_cross_entropy(ctx, logits, labels):
    """Hard-label loss with analytic derivatives and portable ONNX function body."""
    result = ctx.op(_name("HardCrossEntropy", logits.dtype), logits, labels, domain=DOMAIN)
    result.type, result.shape = ir.TensorType(logits.dtype), labels.shape
    return result


def _distill_gradient(ctx, student, teacher):
    axes, zero = ctx.constant([-1]), ctx.scalar(student, 0)
    logq, logp = log_softmax(ctx, student), log_softmax(ctx, teacher)
    q, p = ctx.op("Exp", logq), ctx.op("Exp", logp)
    delta = ctx.op("Sub", student, teacher)
    # Remove a common logit shift before probability-weighted centering.
    anchor = ctx.op("Unsqueeze", ctx.op("Gather", delta, ctx.constant(0), axis=-1), axes)
    relative = ctx.op("Sub", delta, anchor)
    mean = ctx.op("ReduceSum", ctx.op("Mul", p, relative), axes, keepdims=1)
    centered = ctx.op("Sub", relative, mean)
    close = ctx.op(
        "LessOrEqual",
        ctx.op("ReduceMax", ctx.op("Abs", centered), axes, keepdims=1),
        ctx.scalar(student, 0.5),
    )
    small = ctx.op("Where", close, centered, zero)
    small.type = ir.TensorType(student.dtype)
    correction = ctx.op(
        "ReduceSum", ctx.op("Mul", p, _call(ctx, "Expm1", small)), axes, keepdims=1
    )
    correction.type = ir.TensorType(student.dtype)
    ratio = ctx.op("Sub", small, _call(ctx, "Log1p", correction))
    ratio.type = ir.TensorType(student.dtype)
    u = ctx.op("Mul", p, _call(ctx, "Expm1", ratio))
    near = ctx.op("Sub", u, ctx.op("Mul", q, ctx.op("ReduceSum", u, axes, keepdims=1)))
    value = ctx.op("Where", close, near, ctx.op("Sub", q, p))
    mask = _mask(ctx, student, ctx.op("ArgMax", student, axis=-1, keepdims=1))
    others = ctx.op("Where", mask, zero, value)
    return ctx.op(
        "Where", mask, ctx.op("Neg", ctx.op("ReduceSum", others, axes, keepdims=1)), others
    )


def distillation(ctx, student, teacher):
    result = ctx.op(_name("Distillation", student.dtype), student, teacher, domain=DOMAIN)
    result.type = ir.TensorType(student.dtype)
    result.shape = None if student.shape is None else ir.Shape(student.shape[:-1])
    return result


def _function(op, dtype):
    x = ir.val("x", dtype=dtype)
    inputs = [x]
    graph = ir.Graph(inputs, [], nodes=[], opset_imports={"": 18, DOMAIN: 1}, name=op)
    ctx = onnax.AutodiffContext(graph)
    one, zero = ctx.scalar(x, 1), ctx.scalar(x, 0)
    if op == "HardCrossEntropy":
        labels = ir.val("labels", dtype=ir.DataType.INT64)
        graph.inputs.append(labels)
        value = _hard_body(ctx, x, labels)
    elif op == "Log1p":
        u = ctx.op("Add", one, x)
        small = ctx.op("Equal", u, one)
        denominator = ctx.op("Where", small, one, ctx.op("Sub", u, one))
        value = ctx.op(
            "Where", small, x, ctx.op("Mul", ctx.op("Log", u), ctx.op("Div", x, denominator))
        )
    elif op == "Expm1":
        small = ctx.op("LessOrEqual", ctx.op("Abs", x), ctx.scalar(x, 0.5))
        safe = ctx.op("Where", small, x, zero)
        y = ctx.op("Exp", safe)
        rounded = ctx.op("Equal", y, one)
        denominator = ctx.op("Where", rounded, one, ctx.op("Log", y))
        compensated = ctx.op(
            "Where",
            rounded,
            safe,
            ctx.op("Mul", ctx.op("Sub", y, one), ctx.op("Div", safe, denominator)),
        )
        large = ctx.op("Sub", ctx.op("Exp", ctx.op("Where", small, zero, x)), one)
        value = ctx.op("Where", small, compensated, large)
    else:
        teacher = ir.val("teacher", dtype=dtype)
        graph.inputs.append(teacher)
        logq, logp = log_softmax(ctx, x), log_softmax(ctx, teacher)
        value = ctx.op(
            "Neg",
            ctx.op(
                "ReduceSum",
                ctx.op("Mul", ctx.op("Exp", logp), logq),
                ctx.constant([-1]),
                keepdims=0,
            ),
        )
    value.type = ir.TensorType(dtype)
    graph.outputs.append(value)
    return ir.Function(DOMAIN, _name(op, dtype), graph=graph, attributes=[])


def install(model):
    """Attach portable local functions; global rules are installed on import."""
    for dtype in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
        for op in ("Log1p", "Expm1", "Distillation", "HardCrossEntropy"):
            function = _function(op, dtype)
            model.functions[function.identifier()] = function
    model.graph.opset_imports[DOMAIN] = 1


def _register(op, dtype):
    name = _name(op, dtype)

    @onnax.register_vjp(DOMAIN, name, since_version=1, until_version=1)
    def vjp(ctx, node, cotangents, input_indices):
        x = ctx.input(node, 0)
        g = cotangents[0]
        if op == "HardCrossEntropy":
            if list(input_indices) != [0]:
                raise ValueError("HardCrossEntropy differentiates logits only")
            derivative = _hard_gradient(ctx, x, ctx.input(node, 1))
            result = ctx.op("Mul", derivative, ctx.op("Unsqueeze", g, ctx.constant([-1])))
        elif op == "Distillation":
            if list(input_indices) != [0]:
                raise ValueError("Distillation holds teacher logits fixed")
            derivative = _distill_gradient(ctx, x, ctx.input(node, 1))
            result = ctx.op("Mul", derivative, ctx.op("Unsqueeze", g, ctx.constant([-1])))
        else:
            slope = (
                ctx.op("Exp", x)
                if op == "Expm1"
                else ctx.op("Reciprocal", ctx.op("Add", ctx.scalar(x, 1), x))
            )
            result = ctx.op("Mul", g, slope)
        result.type, result.shape = ir.TensorType(dtype), x.shape
        return {0: result}

    @onnax.register_jvp(DOMAIN, name, since_version=1, until_version=1)
    def jvp(ctx, node, tangents):
        x = ctx.input(node, 0)
        if op == "HardCrossEntropy":
            if set(tangents) != {0}:
                raise ValueError("HardCrossEntropy differentiates logits only")
            derivative = _hard_gradient(ctx, x, ctx.input(node, 1))
            result = ctx.op(
                "ReduceSum",
                ctx.op("Mul", derivative, tangents[0]),
                ctx.constant([-1]),
                keepdims=0,
            )
            result.shape = node.outputs[0].shape
        elif op == "Distillation":
            if set(tangents) != {0}:
                raise ValueError("Distillation holds teacher logits fixed")
            derivative = _distill_gradient(ctx, x, ctx.input(node, 1))
            result = ctx.op(
                "ReduceSum",
                ctx.op("Mul", derivative, tangents[0]),
                ctx.constant([-1]),
                keepdims=0,
            )
            result.shape = node.outputs[0].shape
        else:
            slope = (
                ctx.op("Exp", x)
                if op == "Expm1"
                else ctx.op("Reciprocal", ctx.op("Add", ctx.scalar(x, 1), x))
            )
            result = ctx.op("Mul", tangents[0], slope)
            result.shape = x.shape
        result.type = ir.TensorType(dtype)
        return [result]

    @onnax.register_batching(DOMAIN, name, since_version=1, until_version=1)
    def batching(ctx, node):
        if op not in ("Distillation", "HardCrossEntropy"):
            ctx.elementwise(node)
            return
        inputs, mapped = ctx.prepare(node)
        rank = max(
            ctx.rank(v) + (1 if op == "HardCrossEntropy" and i == 1 else 0)
            for i, v in enumerate(inputs)
        )
        if mapped:
            for i, v in enumerate(inputs):
                target_rank = rank - (1 if op == "HardCrossEntropy" and i == 1 else 0)
                node.replace_input_with(i, ctx.align(v, target_rank))
        ctx.register_outputs(node, rank=rank - 1, mapped=mapped)


for _dtype in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
    for _op in ("Log1p", "Expm1", "Distillation", "HardCrossEntropy"):
        _register(_op, _dtype)
