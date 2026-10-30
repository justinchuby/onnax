# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exact FP32 dot-product intervals for the mixed-precision backward graph."""

from fractions import Fraction

import numpy as np

from examples.lora.bf16_bounds import ProductBounds


def integers(values, *, allow_subnormals=False):
    """Encode FP32 values as integer multiples of 2**-149."""
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        raise TypeError("Expected NumPy FP32 arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("Expected finite FP32 values")
    result = np.empty(values.shape, dtype=object)
    bits = values.view(np.uint32)
    for index in np.ndindex(values.shape):
        word = int(bits[index])
        exponent, fraction = (word >> 23) & 255, word & 0x7FFFFF
        if exponent == 0:
            if fraction and not allow_subnormals:
                raise ValueError("FP32 subnormal operands require an input-flush policy")
            value = fraction
        else:
            value = ((1 << 23) + fraction) << (exponent - 1)
        result[index] = -value if word >> 31 else value
    return result


def outward_endpoint(numerator, denominator, *, upper):
    """Enclose a rational in an FP32 endpoint, checking rounding with exact integers."""
    exact = Fraction(int(numerator), int(denominator))
    maximum = Fraction.from_float(float(np.finfo(np.float32).max))
    if abs(exact) > maximum:
        raise ValueError("Accumulation bound reaches FP32 overflow")
    rounded = np.float32(float(exact))
    represented = Fraction.from_float(float(rounded))
    if (upper and represented < exact) or (not upper and represented > exact):
        rounded = np.nextafter(rounded, np.float32(np.inf if upper else -np.inf))
    return rounded


def product_bounds(a, b):
    """Enclose FP32 accumulation with m=2k, u=2**-24 and absolute error 2**-126.

    Each elementary multiply/add obeys |fl(t)-t| <= u*|t|+tiny. The interval
    includes reassociated summations, FMA, gradual underflow and result flushing.
    Operands are normal FP32 or zero; overflow and input flushing are excluded.
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
    left, right = integers(a), integers(b)
    total = left @ right
    magnitude = np.abs(left) @ np.abs(right)
    lower, upper = np.empty(total.shape, np.float32), np.empty(total.shape, np.float32)
    # Products use units 2**-298; tiny/u uses 2**196 of these units.
    for index in np.ndindex(total.shape):
        error = operations * (magnitude[index] + (1 << 196))
        max_fp32_units = ((1 << 24) - 1) << 402
        if magnitude[index] * denominator + error >= max_fp32_units * denominator:
            raise ValueError("Accumulation bound reaches FP32 overflow")
        center = total[index] * denominator
        lower[index] = outward_endpoint(center - error, denominator << 298, upper=False)
        upper[index] = outward_endpoint(center + error, denominator << 298, upper=True)
    return ProductBounds(lower, upper, k, bf16_result=False)
