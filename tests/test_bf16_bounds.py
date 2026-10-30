# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent midpoint, accumulation-order and observed-graph bound checks."""

import itertools
from fractions import Fraction

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora.bf16 import round_bf16
from examples.lora.bf16_bounds import _round_endpoint, product_bounds
from examples.lora.bf16_inspector import ProductInspector, instrument
from examples.lora.mlx_bf16 import gradient_model
from examples.lora.train import problem


def round_fraction(value):
    return _round_endpoint(value.numerator << 266, value.denominator)


def test_rounding_endpoints_resolve_ties_beyond_fp64_precision():
    midpoint = Fraction(1) + Fraction(1, 256)
    offset = Fraction(1, 2**70)
    assert round_fraction(midpoint) == pytest.approx(1)
    assert round_fraction(midpoint - offset) == pytest.approx(1)
    assert round_fraction(midpoint + offset) == pytest.approx(1.0078125)
    assert round_fraction(-midpoint - offset) == pytest.approx(-1.0078125)
    assert round_fraction(-midpoint + offset) == pytest.approx(-1)
    # The next midpoint has an odd lower significand, so the tie rounds up.
    assert round_fraction(midpoint + Fraction(1, 128)) == pytest.approx(1.015625)
    tiny_midpoint = Fraction(1, 2**134)
    assert round_fraction(tiny_midpoint) == pytest.approx(0, abs=0)
    assert round_fraction(tiny_midpoint + Fraction(1, 2**200)) == pytest.approx(
        2**-133, rel=0, abs=0
    )
    with pytest.raises(ValueError, match="BF16 overflow"):
        round_fraction(Fraction(2**128))


def test_midpoint_bound_allows_two_rounding_outcomes_and_rejects_third():
    a = np.array([[1, 2**-8, 2**-26]], np.float32)
    b = np.ones((3, 1), np.float32)
    bounds = product_bounds(a, b)
    np.testing.assert_array_equal(bounds.lower, [[1]])
    np.testing.assert_array_equal(bounds.upper, [[1.0078125]])
    assert bounds.check(np.array([[1]], np.float32))["violations"] == 0
    assert bounds.check(np.array([[1.0078125]], np.float32))["violations"] == 0
    assert bounds.check(np.array([[1.015625]], np.float32))["violations"] == 1


def test_well_separated_dot_has_single_bf16_result():
    bounds = product_bounds(np.ones((1, 16), np.float32), np.ones((16, 1), np.float32))
    np.testing.assert_array_equal(bounds.lower, [[16]])
    np.testing.assert_array_equal(bounds.upper, [[16]])
    assert bounds.check(np.array([[16.125]], np.float32))["violations"] == 1


def test_every_small_accumulation_order_fits_bounds():
    a = np.array([[2**20, 1, -(2**20), -0.5]], np.float32)
    b = np.ones((4, 1), np.float32)
    bounds = product_bounds(a, b)
    for order in itertools.permutations(range(4)):
        total = np.float32(0)
        for index in order:
            total = np.float32(total + a[0, index] * b[index, 0])
        output = round_bf16(np.array([[total]], np.float32))
        assert bounds.check(output)["violations"] == 0


def test_underflow_result_flush_and_invalid_contract_inputs():
    bounds = product_bounds(np.array([[2**-126]], np.float32), np.array([[2**-8]], np.float32))
    assert bounds.check(np.zeros((1, 1), np.float32))["violations"] == 0
    for a, b, error in [
        (np.ones(2, np.float32), np.ones((2, 1), np.float32), "rank-two"),
        (np.empty((0, 2), np.float32), np.ones((2, 1), np.float32), "nonempty"),
        (np.array([[1.1]], np.float32), np.ones((1, 1), np.float32), "rounded"),
        (np.array([[np.inf]], np.float32), np.ones((1, 1), np.float32), "finite"),
        (np.array([[2**-133]], np.float32), np.ones((1, 1), np.float32), "input-flush"),
        (np.array([[2**127]], np.float32), np.array([[4]], np.float32), "FP32 overflow"),
    ]:
        with pytest.raises(ValueError, match=error):
            product_bounds(a, b)
    with pytest.raises(TypeError, match="FP32"):
        product_bounds(np.ones((1, 1)), np.ones((1, 1), np.float32))
    with pytest.raises(ValueError, match="shape"):
        bounds.check(np.zeros((2, 1), np.float32))


def test_instrumented_graph_observes_all_four_forward_products():
    w, c, x, y, _, _, params = problem()
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph = gradient_model(w, c, batch=3)
    before = ir.to_proto(graph).SerializeToString()
    observed, count, taps = instrument(graph)
    assert len(taps) == 4 and count == 4
    assert len(observed.graph.outputs) == 16
    feeds = {"features": x[:3], "target": y[:3], **params}
    original = ReferenceEvaluator(ir.to_proto(graph)).run(None, feeds)
    inspector = ProductInspector(
        graph, session_factory=lambda g: ReferenceEvaluator(ir.to_proto(g))
    )
    report = inspector.check(feeds, original)
    assert report["original_outputs_equal"]
    assert all(not p["violations"] for p in report["products"].values())
    changed = [value.copy() for value in original]
    changed[0] += np.float32(1)
    assert not inspector.check(feeds, changed)["original_outputs_equal"]
    zero_feeds = {
        **feeds,
        "features": np.zeros_like(feeds["features"]),
        "target": np.zeros_like(feeds["target"]),
    }
    zero_result = ReferenceEvaluator(ir.to_proto(graph)).run(None, zero_feeds)
    signed = [value.copy() for value in zero_result]
    signed[0][0, 0] = np.float32(0 if np.signbit(signed[0][0, 0]) else -0.0)
    assert np.array_equal(zero_result[0], signed[0])
    assert not inspector.check(zero_feeds, signed)["original_outputs_equal"]
    assert before == ir.to_proto(graph).SerializeToString()


def test_exact_rounding_matches_independent_fraction_distance_oracle():
    # Enumerate the finite BF16 lattice, then choose by exact rational distance.
    words = np.arange(65536, dtype=np.uint32) << 16
    lattice = np.unique(words.view(np.float32)[np.isfinite(words.view(np.float32))])
    rng = np.random.default_rng(17)
    for exponent in [-133, -126, -20, 0, 30, 120]:
        for sign in [-1, 1]:
            numerator = sign * int(rng.integers(128, 256))
            value = Fraction(numerator) * Fraction(2) ** (exponent - 7) + Fraction(
                sign, 2**200
            )
            center = int(np.searchsorted(lattice, float(value)))
            candidates = lattice[max(0, center - 2) : center + 2]

            def rank(candidate, target=value):
                word = int(np.asarray(candidate).view(np.uint32))
                return abs(Fraction(float(candidate)) - target), (word >> 16) & 1

            expected = min(candidates, key=rank)
            np.testing.assert_array_equal(np.float32(round_fraction(value)), expected)


def test_rejects_invalid_operation_budget_and_final_bf16_overflow():
    one = np.array(1, np.float32)
    a = np.broadcast_to(one, (1, 1 << 23))
    b = np.broadcast_to(one, (1 << 23, 1))
    with pytest.raises(ValueError, match=r"m\*u < 1"):
        product_bounds(a, b)
    a = np.array([[2**127, 2**127]], np.float32)
    b = np.array([[1], [255 / 256]], np.float32)
    with pytest.raises(ValueError, match="BF16 overflow"):
        product_bounds(a, b)
