# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Forward and reverse rules for elementwise operators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import onnx_ir as ir

from onnax._autodiff._common import _BINARY, _FLOATS, _float, _input
from onnax._autodiff._context import _Context
from onnax._rules import _builtin

OPS = frozenset(
    (
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Add",
        "Celu",
        "Cos",
        "Cosh",
        "Div",
        "Elu",
        "Erf",
        "Exp",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "LeakyRelu",
        "Log",
        "Mean",
        "Mul",
        "Neg",
        "Pow",
        "PRelu",
        "Reciprocal",
        "Relu",
        "Selu",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Sub",
        "Sum",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    )
)


def _gelu_partial(ctx: _Context, node: ir.Node, x: ir.Value) -> ir.Value:
    """Differentiate the source GELU approximation, without changing its mode."""
    approximation = node.attributes.get_string("approximate", "none")
    one, half = ctx.scalar(x, 1), ctx.scalar(x, 0.5)
    zero = ctx.scalar(x, 0)
    limit = ctx.op("Where", ctx.op("Greater", x, zero), one, zero)
    if approximation == "none":
        cdf = ctx.op(
            "Mul",
            half,
            ctx.op(
                "Add", one, ctx.op("Erf", ctx.op("Mul", x, ctx.scalar(x, 1 / math.sqrt(2))))
            ),
        )
        density = ctx.op(
            "Mul",
            ctx.scalar(x, 1 / math.sqrt(2 * math.pi)),
            ctx.op("Exp", ctx.op("Mul", ctx.scalar(x, -0.5), ctx.op("Mul", x, x))),
        )
        return ctx.op(
            "Where",
            ctx.op("Equal", density, zero),
            limit,
            ctx.op("Add", cdf, ctx.op("Mul", x, density)),
        )
    if approximation == "tanh":
        square = ctx.op("Mul", x, x)
        scale = ctx.scalar(x, math.sqrt(2 / math.pi))
        argument = ctx.op(
            "Mul",
            scale,
            ctx.op("Add", x, ctx.op("Mul", ctx.scalar(x, 0.044715), ctx.op("Mul", square, x))),
        )
        # Avoid cancellation in 1 - tanh(argument)**2 near saturation. This
        # equivalent sech-squared form retains small FLOAT derivatives.
        positive = ctx.op("GreaterOrEqual", argument, zero)
        magnitude = ctx.op("Where", positive, argument, ctx.op("Neg", argument))
        tail = ctx.op("Exp", ctx.op("Mul", ctx.scalar(x, -2), magnitude))
        denominator = ctx.op("Add", one, tail)
        # (1 + tanh(u))/2 = sigmoid(2u). Use the same stable tail rather
        # than a rounded Tanh output so HVPs retain its small derivative too.
        # The magnitude's positive branch at zero preserves the CDF slope;
        # Abs would assign a zero slope there and lose half the curvature.
        cdf = ctx.op(
            "Where",
            positive,
            ctx.op("Div", one, denominator),
            ctx.op("Div", tail, denominator),
        )
        tanh_partial = ctx.op(
            "Div",
            ctx.op("Mul", ctx.scalar(x, 4), tail),
            ctx.op("Mul", denominator, denominator),
        )
        argument_partial = ctx.op(
            "Mul",
            scale,
            ctx.op("Add", one, ctx.op("Mul", ctx.scalar(x, 3 * 0.044715), square)),
        )
        correction = ctx.op(
            "Mul",
            ctx.op("Mul", x, tanh_partial),
            argument_partial,
        )
        # Once the exponential underflows, select a constant asymptote. This
        # also masks 0*inf intermediates when differentiating saturated tails.
        return ctx.op(
            "Where",
            ctx.op("Equal", tail, zero),
            limit,
            ctx.op("Add", cdf, ctx.op("Mul", half, correction)),
        )
    raise ValueError(
        f"Gelu approximate={approximation!r}; expected 'none' or 'tanh'. "
        "Set the source attribute to the intended supported GELU approximation; "
        "the derivative must use the same mode as the primal computation."
    )


def _partials(ctx: _Context, node: ir.Node, indices: Sequence[int]) -> dict[int, ir.Value]:
    """Elementwise local derivatives; do not evaluate unused partials (notably Pow)."""
    op = node.op_type
    x, y = _input(node), node.outputs[0]
    assert x.dtype in _FLOATS
    one = ctx.scalar(x, 1)
    result = {}
    for index in indices:
        if op == "Mul":
            p = _input(node, 1 - index)
        elif op == "Div":
            b = _input(node, 1)
            p = ctx.op("Div", one, b) if index == 0 else ctx.op("Neg", ctx.op("Div", y, b))
        elif op == "Pow":
            b = _input(node, 1)
            if b.dtype != x.dtype:
                b = ctx.op("Cast", b, to=x.dtype)
            if index == 0:
                # Guard only 0**0, preserving mixed exponent derivatives at b=0
                # for positive bases. A blanket b=0 branch loses that dependence.
                zero = ctx.scalar(x, 0)
                at_origin = ctx.op("And", ctx.op("Equal", x, zero), ctx.op("Equal", b, zero))
                safe_b = ctx.op("Where", at_origin, one, b)
                p = ctx.op("Mul", b, ctx.op("Pow", x, ctx.op("Sub", safe_b, one)))
            else:
                p = ctx.op("Mul", y, ctx.op("Log", x))
        elif op == "PRelu":
            zero = ctx.scalar(x, 0)
            positive = ctx.op("Greater", x, zero)
            p = (
                ctx.op("Where", positive, one, _input(node, 1))
                if index == 0
                else ctx.op("Where", positive, zero, x)
            )
        elif op == "Sign":
            # Zero almost everywhere; choose zero at the discontinuity as well.
            p = ctx.scalar(x, 0)
        elif op == "Abs":
            p = ctx.op("Sign", x)
        elif op == "Exp":
            p = y
        elif op == "Gelu":
            p = _gelu_partial(ctx, node, x)
        elif op in ("HardSigmoid", "HardSwish"):
            # Attribute defaults and HardSwish's function-body alpha are FLOAT
            # literals in ONNX, including when the input tensor is DOUBLE.
            alpha = (
                _float(node, "alpha", 0.20000000298023224)
                if op == "HardSigmoid"
                else 0.1666666716337204
            )
            beta = _float(node, "beta", 0.5) if op == "HardSigmoid" else 0.5
            zero = ctx.scalar(x, 0)
            affine = ctx.op("Add", ctx.op("Mul", ctx.scalar(x, alpha), x), ctx.scalar(x, beta))
            interior = ctx.op(
                "And", ctx.op("Greater", affine, zero), ctx.op("Less", affine, one)
            )
            if op == "HardSigmoid":
                p = ctx.op("Where", interior, ctx.scalar(x, alpha), zero)
            else:
                middle = ctx.op(
                    "Add", ctx.op("Mul", ctx.scalar(x, 2 * alpha), x), ctx.scalar(x, beta)
                )
                saturated = ctx.op("Where", ctx.op("GreaterOrEqual", affine, one), one, zero)
                p = ctx.op("Where", interior, middle, saturated)
        elif op == "Log":
            p = ctx.op("Div", one, x)
        elif op == "Sqrt":
            p = ctx.op("Div", ctx.scalar(x, 0.5), y)
        elif op == "Reciprocal":
            p = ctx.op("Neg", ctx.op("Mul", y, y))
        elif op in ("Asin", "Acos"):
            denominator = ctx.op("Sqrt", ctx.op("Sub", one, ctx.op("Mul", x, x)))
            p = ctx.op("Div", ctx.scalar(x, -1) if op == "Acos" else one, denominator)
        elif op == "Acosh":
            # Separate reciprocal roots avoid overflow in x*x or the product
            # (x-1)*(x+1), while retaining accuracy next to the domain boundary.
            lower = ctx.op("Reciprocal", ctx.op("Sqrt", ctx.op("Sub", x, one)))
            upper = ctx.op("Reciprocal", ctx.op("Sqrt", ctx.op("Add", x, one)))
            p = ctx.op("Mul", lower, upper)
        elif op == "Atanh":
            # Keep x*x so higher-order derivatives retain tiny curvature near
            # zero instead of subtracting the rounded slopes of (1-x)*(1+x).
            p = ctx.op("Reciprocal", ctx.op("Sub", one, ctx.op("Mul", x, x)))
        elif op in ("Atan", "Asinh"):
            # Evaluate inverse quadratic norms with a bounded operand. Both Where
            # branches execute, so also protect the inactive reciprocal at zero.
            large = ctx.op("Greater", ctx.op("Abs", x), one)
            reciprocal = ctx.op("Reciprocal", ctx.op("Where", large, x, one))
            bounded = ctx.op("Where", large, reciprocal, x)
            square = ctx.op("Mul", bounded, bounded)
            denominator = ctx.op("Add", one, square)
            if op == "Asinh":
                inverse = ctx.op("Reciprocal", ctx.op("Sqrt", denominator))
                norm_scale = ctx.op("Abs", reciprocal)
            else:
                inverse = ctx.op("Reciprocal", denominator)
                norm_scale = square
            p = ctx.op("Where", large, ctx.op("Mul", norm_scale, inverse), inverse)
        elif op == "Sin":
            p = ctx.op("Cos", x)
        elif op == "Cos":
            p = ctx.op("Neg", ctx.op("Sin", x))
        elif op == "Tan":
            p = ctx.op("Add", one, ctx.op("Mul", y, y))
        elif op == "Sinh":
            p = ctx.op("Cosh", x)
        elif op == "Cosh":
            p = ctx.op("Sinh", x)
        elif op in ("Tanh", "Sigmoid"):
            # Avoid subtracting rounded saturated outputs from one. Selecting
            # the positive branch at zero also preserves higher-order behavior.
            positive = ctx.op("GreaterOrEqual", x, ctx.scalar(x, 0))
            magnitude = ctx.op("Where", positive, x, ctx.op("Neg", x))
            scale = -2 if op == "Tanh" else -1
            tail = ctx.op("Exp", ctx.op("Mul", ctx.scalar(x, scale), magnitude))
            denominator = ctx.op("Add", one, tail)
            numerator = ctx.op("Mul", ctx.scalar(x, 4), tail) if op == "Tanh" else tail
            p = ctx.op("Div", numerator, ctx.op("Mul", denominator, denominator))
        elif op in ("Relu", "LeakyRelu", "ThresholdedRelu"):
            negative = _float(node, "alpha", 0.01) if op == "LeakyRelu" else 0.0
            threshold = _float(node, "alpha", 1.0) if op == "ThresholdedRelu" else 0.0
            p = ctx.op(
                "Where",
                ctx.op("Greater", x, ctx.scalar(x, threshold)),
                one,
                ctx.scalar(x, negative),
            )
        elif op == "Celu":
            alpha = _float(node, "alpha", 1.0)
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError(
                    f"Celu differentiation requires finite alpha > 0; got {alpha}. "
                    "Use a positive shape parameter."
                )
            positive = ctx.op("Greater", x, ctx.scalar(x, 0))
            negative_x = ctx.op("Where", positive, ctx.scalar(x, 0), x)
            slope = ctx.op("Exp", ctx.op("Div", negative_x, ctx.scalar(x, alpha)))
            p = ctx.op("Where", positive, one, slope)
        elif op in ("Elu", "Selu"):
            # ONNX stores attribute defaults as float32, even for DOUBLE inputs.
            alpha = _float(node, "alpha", 1.0 if op == "Elu" else 1.6732631921768188)
            gamma = 1.0 if op == "Elu" else _float(node, "gamma", 1.0507010221481323)
            positive = ctx.op("Greater", x, ctx.scalar(x, 0))
            # Where evaluates both branches. Clamp the exponential's input so
            # large positive primals cannot overflow the inactive branch or HVP.
            negative_x = ctx.op("Where", positive, ctx.scalar(x, 0), x)
            slope = ctx.op("Mul", ctx.scalar(x, alpha * gamma), ctx.op("Exp", negative_x))
            p = ctx.op("Where", positive, ctx.scalar(x, gamma), slope)
        elif op == "Softplus":
            p = ctx.op("Sigmoid", x)
        elif op == "Softsign":
            denom = ctx.op("Add", one, ctx.op("Abs", x))
            # Squaring the reciprocal avoids overflow in both the slope and
            # the denominator generated by forward-over-reverse differentiation.
            inverse = ctx.op("Reciprocal", denom)
            p = ctx.op("Mul", inverse, inverse)
        elif op == "Erf":
            p = ctx.op(
                "Mul",
                ctx.scalar(x, 2 / math.sqrt(math.pi)),
                ctx.op("Exp", ctx.op("Neg", ctx.op("Mul", x, x))),
            )
            # The exponential can underflow after x*x overflows. Mask its
            # differentiated inactive branch so 0*inf cannot poison the HVP.
            zero = ctx.scalar(x, 0)
            p = ctx.op("Where", ctx.op("Equal", p, zero), zero, p)
        else:
            raise NotImplementedError(
                f"No elementwise derivative for {op}, input index {index}. "
                "Correct the private registry dispatch or implement and validate the "
                "missing partial derivative; do not silently substitute zero."
            )
        result[index] = p
    return result


@_builtin("jvp", OPS)
def jvp(ctx: _Context, node: ir.Node, d: Mapping[int, ir.Value]) -> list[ir.Value]:
    op = node.op_type
    if op == "Identity" or (op in {"Sum", "Mean"} and len(node.inputs) == 1):
        return [d[0]]
    if op == "Neg":
        return [ctx.op("Neg", d[0])]
    if op in OPS:
        if op == "Sub":
            if 0 in d and 1 in d:
                result = ctx.op("Sub", d[0], d[1])
            else:
                result = d[0] if 0 in d else ctx.op("Neg", d[1])
        elif op in {"Add", "Sum"}:
            result = ctx.sum(list(d.values()))
        elif op == "Mean":
            scale = ctx.scalar(_input(node), 1 / len(node.inputs))
            # Scale before summing to avoid overflowing a finite mean tangent.
            result = ctx.sum([ctx.op("Mul", tangent, scale) for tangent in d.values()])
        else:
            partials = _partials(ctx, node, tuple(d))
            result = ctx.sum([ctx.op("Mul", d[i], partials[i]) for i in d])
        return [
            ctx.op("Expand", result, ctx.op("Shape", node.outputs[0]))
            if op in _BINARY | {"Sum", "Mean"}
            else result
        ]
    raise NotImplementedError(
        f"No JVP rule for {op} in the elementwise family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero tangents."
    )


@_builtin("vjp", OPS)
def vjp(
    ctx: _Context, node: ir.Node, gs: Sequence[ir.Value], indices: Sequence[int]
) -> dict[int, ir.Value]:
    op, g = node.op_type, gs[0]
    if op == "Identity" or (op in {"Sum", "Mean"} and len(node.inputs) == 1):
        return {0: g}
    if op == "Neg":
        return {0: ctx.op("Neg", g)}
    if op in {"Add", "Sub", "Sum", "Mean"}:
        # Share the cotangent (and Mean scaling) before undoing broadcasting.
        # Sub only negates the second input's contribution.
        contribution = (
            ctx.op("Mul", g, ctx.scalar(_input(node), 1 / len(node.inputs)))
            if op == "Mean"
            else g
        )
        return {
            i: ctx.unbroadcast(
                ctx.op("Neg", contribution) if op == "Sub" and i == 1 else contribution,
                _input(node, i),
                broadcasted=node.outputs[0],
            )
            for i in indices
        }
    if op in OPS:
        partials = _partials(ctx, node, indices)
        result = {}
        for i in indices:
            contribution = ctx.op("Mul", g, partials[i])
            # PRelu only broadcasts slope to data, so its data pullback already
            # has the exact input shape, even when dimension sizes are unknown.
            result[i] = (
                ctx.unbroadcast(contribution, _input(node, i), broadcasted=node.outputs[0])
                if op in _BINARY | {"Sum", "Mean"} or (op == "PRelu" and i == 1)
                else contribution
            )
        return result
    raise NotImplementedError(
        f"No VJP rule for {op} in the elementwise family. "
        "Correct the private registry dispatch or implement and validate the missing rule; "
        "unsupported active paths cannot be replaced with zero cotangents."
    )
