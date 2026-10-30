# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exact-integer BF16 dot-product bounds for the native precision experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProductBounds:
    lower: np.ndarray
    upper: np.ndarray
    inner_dimension: int
    bf16_result: bool = True

    def check(self, actual):
        """Compare observed results with the product interval."""
        if self.bf16_result:
            _integers(actual, allow_subnormals=True)
        elif not isinstance(actual, np.ndarray) or actual.dtype != np.float32:
            raise TypeError("Expected NumPy FP32 output")
        elif not np.all(np.isfinite(actual)):
            raise ValueError("Expected finite FP32 output")
        if actual.shape != self.lower.shape:
            raise ValueError("Product output shape mismatch")
        bad = (actual < self.lower) | (actual > self.upper)
        report = {
            "elements": actual.size,
            "violations": int(np.count_nonzero(bad)),
            "ambiguous_elements": int(np.count_nonzero(self.lower != self.upper)),
            "max_interval_width": float(np.max(self.upper.astype(np.float64) - self.lower)),
        }
        if np.any(bad):
            index = tuple(np.argwhere(bad)[0])
            report["first_violation"] = {
                "index": [int(i) for i in index],
                "actual": float(actual[index]),
                "lower": float(self.lower[index]),
                "upper": float(self.upper[index]),
            }
        return report


def _integers(values, *, allow_subnormals=False):
    """Encode BF16 values as exact integer multiples of 2**-133."""
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        raise TypeError("Expected BF16-expanded NumPy FP32 arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("Expected finite BF16 values")
    bits = values.view(np.uint32)
    if np.any(bits & np.uint32(0xFFFF)):
        raise ValueError("Values must already be rounded to BF16")
    result = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        word = int(bits[index])
        exponent, fraction = (word >> 23) & 255, (word >> 16) & 127
        if exponent == 0:
            if fraction and not allow_subnormals:
                raise ValueError("BF16 subnormal operands require an input-flush policy")
            integer = fraction
        else:
            integer = (128 + fraction) << (exponent - 1)
        result[index] = -integer if word >> 31 else integer
    return result


def _round_endpoint(numerator, denominator):
    """Round (numerator / denominator) * 2**-266 directly to nearest-even BF16.

    All comparisons and tie decisions use integers, avoiding FP64/FP32 double
    rounding when an interval endpoint lies near a BF16 midpoint.
    """
    if numerator == 0:
        return 0.0
    sign = -1 if numerator < 0 else 1
    numerator = abs(numerator)
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        below = numerator < denominator << exponent
    else:
        below = numerator << -exponent < denominator
    exponent -= int(below)
    spacing = max(exponent - 266 - 7, -133)
    shift = -266 - spacing
    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient & 1):
        quotient += 1
    value = sign * math.ldexp(quotient, spacing)
    if abs(value) >= 2.0**128:
        raise ValueError("Bound reaches BF16 overflow")
    return value


def product_bounds(a, b):
    """Bound BF16 output under FP32 accumulation and final nearest-even rounding.

    For k products, m=2k and u=2**-24, use
    E=(m*u*S + m*tiny)/(1-m*u), S=sum(abs(a_i*b_i)), tiny=2**-126.
    The absolute term covers gradual underflow or flushing arithmetic results
    to zero. Inputs are finite normal BF16 values or zero. Each elementary
    operation obeys |fl(t)-t| <= u*|t|+tiny; at most 2k such operations occur
    on the dot-product tree. Overflow and input flushing are excluded.

    Exact integer products/sums and rational endpoints keep validation error
    out of the bound. The result describes one product at its observed inputs.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible rank-two operands")
    if min(*a.shape, *b.shape) < 1:
        raise ValueError("Expected nonempty matrices")
    k = a.shape[1]
    operations = 2 * k
    denominator = (1 << 24) - operations
    if denominator <= 0:
        raise ValueError("Inner dimension exceeds the bound's m*u < 1 condition")
    left, right = _integers(a), _integers(b)
    total = left @ right
    magnitude = np.abs(left) @ np.abs(right)
    lower = np.empty(total.shape, np.float32)
    upper = np.empty(total.shape, np.float32)
    # Dot products use units 2**-266. tiny/u in those units is 2**164.
    tiny_over_u = 1 << 164
    max_fp32 = ((1 << 24) - 1) << 370
    for index in np.ndindex(total.shape):
        absolute = magnitude[index]
        error_numerator = operations * (absolute + tiny_over_u)
        if absolute * denominator + error_numerator >= max_fp32 * denominator:
            raise ValueError("Accumulation bound reaches FP32 overflow")
        center = total[index] * denominator
        lower[index] = _round_endpoint(center - error_numerator, denominator)
        upper[index] = _round_endpoint(center + error_numerator, denominator)
    return ProductBounds(lower, upper, k)
