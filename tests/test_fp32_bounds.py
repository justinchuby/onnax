# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exact rational checks for mixed-precision backward accumulation intervals."""

import itertools
from fractions import Fraction

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora.bf16_inspector import ProductInspector
from examples.lora.fp32_bounds import integers, outward_endpoint, product_bounds
from examples.lora.mlx_bf16 import gradient_model
from examples.lora.train import problem


def test_outward_endpoints_resolve_sub_fp64_offsets():
    for sign in (-1, 1):
        for exact in (Fraction(1), Fraction(1, 2**149), Fraction(1, 2**150)):
            for offset in (-Fraction(1, 2**220), Fraction(0), Fraction(1, 2**220)):
                value = sign * (exact + offset)
                lower = outward_endpoint(value.numerator, value.denominator, upper=False)
                upper = outward_endpoint(value.numerator, value.denominator, upper=True)
                assert Fraction.from_float(float(lower)) <= value
                assert Fraction.from_float(float(upper)) >= value
                assert lower == upper or np.nextafter(lower, np.float32(np.inf)) == upper


def test_integer_encoding_is_exact_for_normal_and_subnormal_values():
    values = np.array([0, -0.0, 1.1, -3.7, 2**-149, 2**-126, 2**100], np.float32)
    encoded = integers(values, allow_subnormals=True)
    for value, integer in zip(values, encoded, strict=True):
        assert Fraction(int(integer), 2**149) == Fraction.from_float(float(value))
    with pytest.raises(ValueError, match="input-flush"):
        integers(values)


def test_accumulation_orders_and_corrupted_result():
    a = np.array([[2**25, 1.1, -(2**25), -0.3]], np.float32)
    b = np.ones((4, 1), np.float32)
    bounds = product_bounds(a, b)
    for order in itertools.permutations(range(4)):
        total = np.float32(0)
        for i in order:
            total = np.float32(total + a[0, i] * b[i, 0])
        assert bounds.check(np.array([[total]], np.float32))["violations"] == 0
    outside = np.nextafter(bounds.upper, np.float32(np.inf))
    assert bounds.check(outside)["violations"] == 1
    with pytest.raises(ValueError, match="finite"):
        bounds.check(np.array([[np.nan]], np.float32))


def test_underflow_and_overflow_contract():
    bounds = product_bounds(
        np.array([[2**-126]], np.float32), np.array([[2**-20]], np.float32)
    )
    assert bounds.check(np.array([[0]], np.float32))["violations"] == 0
    with pytest.raises(ValueError, match="overflow"):
        product_bounds(np.array([[2**127]], np.float32), np.array([[4]], np.float32))


def test_every_forward_and_backward_product_in_exported_graph():
    w, c, x, y, _, _, params = problem()
    params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph = gradient_model(w, c, batch=3)
    before = ir.to_proto(graph).SerializeToString()

    def factory(g):
        return ReferenceEvaluator(ir.to_proto(g))

    feeds = {"features": x[:3], "target": y[:3], **params}
    original = factory(graph).run(None, feeds)
    inspector = ProductInspector(graph, session_factory=factory, include_fp32=True)
    report = inspector.check(feeds, original)
    assert report["original_outputs_equal"]
    assert len(report["products"]) == sum(n.op_type == "MatMul" for n in graph.graph)
    assert {row["dtype"] for row in report["products"].values()} == {"FLOAT", "BFLOAT16"}
    assert all(row["violations"] == 0 for row in report["products"].values())
    assert before == ir.to_proto(graph).SerializeToString()


def test_cancellation_overflow_is_rejected_even_when_exact_sum_is_zero():
    a = np.array([[2**127, 2**127, -(2**127), -(2**127)]], np.float32)
    with pytest.raises(ValueError, match="overflow"):
        product_bounds(a, np.ones((4, 1), np.float32))
