# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Outward FP64 interval arithmetic for the LoRA precision experiment."""

from dataclasses import dataclass

import numpy as np


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


@dataclass(frozen=True)
class Interval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lower, upper = np.broadcast_arrays(
            np.asarray(self.lower, np.float64), np.asarray(self.upper, np.float64)
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Interval endpoints must be finite")
        if np.any(lower > upper):
            raise ValueError("Interval endpoints must be ordered")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value):
        return cls(value, value)

    def __add__(self, other):
        other = interval(other)
        return Interval(down(self.lower + other.lower), up(self.upper + other.upper))

    def __neg__(self):
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other):
        return self + -interval(other)

    def __mul__(self, other):
        other = interval(other)
        products = np.array(
            np.broadcast_arrays(
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            )
        )
        return Interval(down(np.min(products, axis=0)), up(np.max(products, axis=0)))

    def __truediv__(self, other):
        other = interval(other)
        if np.any((other.lower <= 0) & (other.upper >= 0)):
            raise ValueError("Interval division crosses zero")
        reciprocal = Interval(down(1 / other.upper), up(1 / other.lower))
        return self * reciprocal

    def square(self):
        low = np.minimum(self.lower * self.lower, self.upper * self.upper)
        high = np.maximum(self.lower * self.lower, self.upper * self.upper)
        low = np.where((self.lower <= 0) & (self.upper >= 0), 0, down(low))
        return Interval(low, up(high))

    def transpose(self, axes):
        return Interval(self.lower.transpose(axes), self.upper.transpose(axes))

    def reshape(self, shape):
        return Interval(self.lower.reshape(shape), self.upper.reshape(shape))

    def check(self, actual):
        actual = np.asarray(actual)
        if actual.shape != self.lower.shape or not np.all(np.isfinite(actual)):
            raise ValueError("Expected a finite observation with the interval shape")
        bad = (actual < self.lower) | (actual > self.upper)
        width = self.upper - self.lower
        result = {
            "elements": int(actual.size),
            "violations": int(np.count_nonzero(bad)),
            "max_width": float(np.max(width)),
            "mean_width": float(np.mean(width)),
        }
        if np.any(bad):
            index = tuple(np.argwhere(bad)[0])
            result["first_violation"] = {
                "index": [int(i) for i in index],
                "actual": float(actual[index]),
                "lower": float(self.lower[index]),
                "upper": float(self.upper[index]),
            }
        return result


def interval(value):
    return value if isinstance(value, Interval) else Interval.point(value)


def fp32(value, *, operations=1, magnitude=None):
    """Expand an exact-result interval under the FP32 elementary-operation model."""
    if operations < 1 or operations * 2**-24 >= 1:
        raise ValueError("FP32 operation budget requires 0 < m*u < 1")
    magnitude = (
        np.maximum(np.abs(value.lower), np.abs(value.upper))
        if magnitude is None
        else magnitude
    )
    denominator = down(1 - operations * 2**-24)
    gamma = up(operations * 2**-24 / denominator)
    absolute = up(operations * 2**-126 / denominator)
    error = up(up(gamma * magnitude) + absolute)
    if np.any(up(magnitude + error) > np.finfo(np.float32).max):
        raise ValueError("FP32 interval reaches overflow")
    return Interval(down(value.lower - error), up(value.upper + error))


def elementwise(a, b, op, *, same_input=False):
    if op == "Add":
        exact = a + b
    elif op == "Sub":
        exact = a - b
    elif op == "Mul":
        exact = a.square() if same_input else a * b
    elif op == "Div":
        exact = a / b
    else:
        raise ValueError(f"Unsupported elementary interval operation: {op}")
    return fp32(exact)


def _point_dot(a, b):
    """Enclose exact FP64-input dot products, including reference accumulation error."""
    operations = 2 * a.shape[1]
    denominator = down(1 - operations * 2**-53)
    if denominator <= 0:
        raise ValueError("FP64 reference operation budget is too large")
    gamma = up(operations * 2**-53 / denominator)
    absolute = up(operations * 2**-1022 / denominator)
    observed = a @ b
    magnitude = up(up(np.abs(a) @ np.abs(b) + absolute) / down(1 - gamma))
    error = up(up(gamma * magnitude) + absolute)
    return Interval(down(observed - error), up(observed + error))


def exact_matmul(a, b):
    """Propagate interval operand uncertainty through an exact real matrix product."""
    if a.lower.ndim != 2 or b.lower.ndim != 2 or a.lower.shape[1] != b.lower.shape[0]:
        raise ValueError("Expected compatible rank-two intervals")
    if min(*a.lower.shape, *b.lower.shape) < 1:
        raise ValueError("Expected nonempty matrices")
    ac = np.clip(a.lower / 2 + a.upper / 2, a.lower, a.upper)
    bc = np.clip(b.lower / 2 + b.upper / 2, b.lower, b.upper)
    ar = up(np.maximum(up(ac - a.lower), up(a.upper - ac)))
    br = up(np.maximum(up(bc - b.lower), up(b.upper - bc)))
    center = _point_dot(ac, bc)
    radius = up(
        up(_point_dot(np.abs(ac), br).upper + _point_dot(ar, np.abs(bc)).upper)
        + _point_dot(ar, br).upper
    )
    return Interval(down(center.lower - radius), up(center.upper + radius))


def matmul(a, b):
    result = exact_matmul(a, b)
    am = np.maximum(np.abs(a.lower), np.abs(a.upper))
    bm = np.maximum(np.abs(b.lower), np.abs(b.upper))
    return fp32(result, operations=2 * a.lower.shape[1], magnitude=_point_dot(am, bm).upper)


def mean(value):
    count = value.lower.size
    summed = exact_matmul(value.reshape((1, count)), Interval.point(np.ones((count, 1))))
    real = (summed / count).reshape(())
    # Sum magnitudes rather than magnitude of a cancelling sum.
    absolute = np.maximum(np.abs(value.lower), np.abs(value.upper))
    magnitude = _point_dot(absolute.reshape((1, count)), np.ones((count, 1))).upper
    # Bound a sum-first reduction's intermediates before dividing by N.
    fp32(Interval.point(magnitude), operations=2 * count, magnitude=magnitude)
    magnitude = up(magnitude / count).reshape(())
    return fp32(real, operations=2 * count + 2, magnitude=magnitude)


def _bf16_round(value):
    """Round FP64 endpoints directly to nearest-even BF16, avoiding double rounding."""
    _, exponent = np.frexp(np.abs(value))
    spacing = np.ldexp(np.ones_like(value), np.maximum(exponent - 8, -133))
    rounded = np.rint(value / spacing) * spacing
    if np.any(np.abs(rounded) >= 2**128):
        raise ValueError("BF16 interval reaches overflow")
    return rounded


def bf16(value):
    return Interval(_bf16_round(value.lower), _bf16_round(value.upper))


def _positive_expm1(value):
    """Enclose exp(value)-1 using a positive Taylor sum and a geometric remainder."""
    if np.any(value < 0) or np.any(value > 80):
        raise ValueError("Nonlinear interval reference requires arguments in [-80, 80]")
    t = Interval.point(value / 256)
    term, total = Interval.point(np.ones_like(value)), Interval.point(np.zeros_like(value))
    for n in range(1, 25):
        term = (term * t) / n
        total = total + term
    next_term = (term * t) / 25
    tail = next_term / (Interval.point(1) - t / 26)
    result = Interval(np.maximum(total.lower, 0), (total + tail).upper)
    for _ in range(8):
        result = result * (result + 2)
    return Interval(np.maximum(result.lower, 0), result.upper)


def exp_point(value):
    positive = _positive_expm1(np.abs(value)) + 1
    inverse = Interval.point(1) / positive
    lower = np.where(value < 0, inverse.lower, positive.lower)
    upper = np.where(value < 0, inverse.upper, positive.upper)
    return Interval(np.where(value == 0, 1, lower), np.where(value == 0, 1, upper))


def tanh_point(value):
    q = _positive_expm1(2 * np.abs(value))
    ratio = q / (q + 2)
    lower = np.where(value < 0, -ratio.upper, ratio.lower)
    upper = np.where(value < 0, -ratio.lower, ratio.upper)
    return Interval(np.maximum(lower, -1), np.minimum(upper, 1))


def nonlinear(value, op, *, ulps=4):
    """Apply monotonic exp/tanh with an explicit FP32 implementation-error budget."""
    if isinstance(ulps, bool) or not isinstance(ulps, int) or ulps < 0:
        raise ValueError("Nonlinear ULP budget must be a nonnegative integer")
    factory = {"Exp": exp_point, "Tanh": tanh_point}[op]
    real = Interval(factory(value.lower).lower, factory(value.upper).upper)
    maximum = np.maximum(np.abs(real.lower), np.abs(real.upper))
    _, exponent = np.frexp(maximum)
    spacing = np.ldexp(np.ones_like(maximum), np.maximum(exponent - 24, -149))
    error = up(ulps * spacing + 2**-126)
    return Interval(down(real.lower - error), up(real.upper + error))
