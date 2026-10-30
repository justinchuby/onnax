# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent rational/Decimal references and full mixed-precision graph checks."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

from examples.lora import intervals as bounds
from examples.lora.bf16 import reference
from examples.lora.bf16_bounds import _round_endpoint
from examples.lora.bf16_composed import GraphInspector, propagate
from examples.lora.mlx_bf16 import gradient_model
from examples.lora.train import problem


def fraction(value):
    return Fraction.from_float(float(value))


def test_elementary_operations_enclose_exact_endpoint_arithmetic():
    rng = np.random.default_rng(600)
    for _ in range(30):
        a, b = (np.sort(rng.normal(size=2)) for _ in range(2))
        ia, ib = bounds.Interval(*a), bounds.Interval(*b)
        for operation, exact in (
            (ia + ib, lambda x, y: x + y),
            (ia - ib, lambda x, y: x - y),
            (ia * ib, lambda x, y: x * y),
        ):
            for x in a:
                for y in b:
                    value = exact(fraction(x), fraction(y))
                    assert fraction(operation.lower) <= value <= fraction(operation.upper)
        positive = bounds.Interval(1, 3)
        quotient = ia / positive
        for x in a:
            for y in (1, 3):
                assert fraction(quotient.lower) <= fraction(x) / y <= fraction(quotient.upper)
    with pytest.raises(ValueError, match="zero"):
        ia / bounds.Interval(-1, 1)


def test_matrix_operand_uncertainty_includes_all_endpoint_products():
    a = bounds.Interval(np.array([[1.0, -3.0]]), np.array([[2.0, -1.0]]))
    b = bounds.Interval(np.array([[-2.0], [1.0]]), np.array([[1.0], [4.0]]))
    result = bounds.exact_matmul(a, b)
    import itertools

    for av in itertools.product(*zip(a.lower.flat, a.upper.flat, strict=True)):
        for bv in itertools.product(*zip(b.lower.flat, b.upper.flat, strict=True)):
            exact = sum((fraction(x) * fraction(y) for x, y in zip(av, bv, strict=True)))
            assert fraction(result.lower[0, 0]) <= exact <= fraction(result.upper[0, 0])
    rounded = bounds.matmul(a, b)
    assert rounded.lower <= result.lower
    assert rounded.upper >= result.upper


@pytest.mark.parametrize("op", ["Exp", "Tanh"])
def test_nonlinear_taylor_intervals_against_independent_decimal(op):
    values = np.array([-39.0, -10.0, -0.1, -(2**-100), 0.0, 2**-100, 0.1, 10.0, 39.0])
    factory = bounds.exp_point if op == "Exp" else bounds.tanh_point
    result = factory(values)
    with localcontext() as ctx:
        ctx.prec = 120
        for i, value in enumerate(values):
            exact_input = Decimal.from_float(float(value))
            if op == "Exp":
                exact = exact_input.exp()
            else:
                exponential = (2 * exact_input).exp()
                exact = (exponential - 1) / (exponential + 1)
            if value == 0 and op == "Exp":
                assert result.lower[i] == result.upper[i] == 1
            else:
                assert Decimal.from_float(float(result.lower[i])) <= exact.next_minus()
                assert Decimal.from_float(float(result.upper[i])) >= exact.next_plus()
    interval = bounds.nonlinear(bounds.Interval(values - 0.01, values + 0.01), op)
    assert np.all(interval.lower <= factory(values - 0.01).lower)
    assert np.all(interval.upper >= factory(values + 0.01).upper)
    with pytest.raises(ValueError, match="arguments"):
        factory(np.array([100.0]))


def test_direct_bf16_rounding_resolves_midpoints_without_fp32_double_rounding():
    midpoint = 1 + 2**-8
    values = np.array(
        [np.nextafter(midpoint, -np.inf), midpoint, np.nextafter(midpoint, np.inf)]
    )
    actual = bounds.bf16(bounds.Interval.point(values)).lower
    expected = []
    for value in values:
        exact = fraction(value)
        expected.append(_round_endpoint(exact.numerator << 266, exact.denominator))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [1, 1, 1.0078125])


@pytest.mark.parametrize("nonzero", [False, True])
def test_complete_exported_graph_and_corruption_detection(nonzero):
    w, c, x, y, _, _, params = problem()
    if nonzero:
        params["lora.B"] = np.full_like(params["lora.B"], 0.05)
    graph = gradient_model(w, c, batch=3)
    before = ir.to_proto(graph).SerializeToString()
    feeds = {"features": x[:3], "target": y[:3], **params}

    def factory(g):
        return ReferenceEvaluator(ir.to_proto(g))

    original = factory(graph).run(None, feeds)
    inspector = GraphInspector(graph, session_factory=factory)
    same = reference(x[:3], y[:3], params, w, c)
    report = inspector.check(feeds, original, reference=[*same[:2], *same[2].values()])
    assert len(report["reference_outputs"]) == 4
    assert report["all_pass"] and report["original_outputs_equal"]
    assert len(report["outputs"]) == 4
    assert len(report["nonlinear_budgets"]) == 2
    enclosures = propagate(graph, feeds)
    for value, actual in zip(graph.graph.outputs, original, strict=True):
        assert enclosures[value.name].check(actual)["violations"] == 0
        assert enclosures[value.name].check(actual + np.float32(1))["violations"] > 0
    changed = [value.copy() for value in original]
    changed[0] += np.float32(1)
    assert not inspector.check(feeds, changed)["all_pass"]
    assert ir.to_proto(graph).SerializeToString() == before


def test_local_nonlinear_budget_detects_corrupted_observation():
    w, c, x, y, _, _, params = problem()
    graph = gradient_model(w, c, batch=3)

    def factory(g):
        return ReferenceEvaluator(ir.to_proto(g))

    feeds = {"features": x[:3], "target": y[:3], **params}
    original = factory(graph).run(None, feeds)
    inspector = GraphInspector(graph, session_factory=factory)
    executor = inspector.executor
    name = next(n.outputs[0].name for n in graph.graph if n.op_type == "Exp")
    position = inspector.names.index(name)

    class Corrupted:
        def run(self, *args):
            values = executor.run(*args)
            values[position] = values[position] + np.float32(1)
            return values

    inspector.executor = Corrupted()
    report = inspector.check(feeds, original)
    assert report["original_outputs_equal"]
    assert report["nonlinear_budgets"][name]["violations"] > 0
    assert not report["all_pass"]


def test_invalid_domains_inputs_and_budgets_fail_explicitly():
    w, c, x, y, _, _, params = problem()
    graph = gradient_model(w, c, batch=3)
    feeds = {"features": x[:3], "target": y[:3], **params}
    with pytest.raises(ValueError, match="match"):
        propagate(graph, {})
    with pytest.raises(ValueError, match="FP32 input"):
        propagate(graph, {**feeds, "target": y[:3].astype(np.float64)})
    first = next(iter(graph.graph))
    first.domain = "custom"
    with pytest.raises(ValueError, match="standard-domain"):
        propagate(graph, feeds)
    with pytest.raises(ValueError, match="budget"):
        bounds.nonlinear(bounds.Interval.point(1), "Exp", ulps=-1)
    with pytest.raises(ValueError, match="overflow"):
        bounds.bf16(bounds.Interval.point(2.0**128))


def test_operand_policy_rejects_subnormal_even_inside_a_composed_interval():
    w, c, x, y, _, _, params = problem()
    graph = gradient_model(w, c, batch=3)

    def factory(g):
        return ReferenceEvaluator(ir.to_proto(g))

    feeds = {"features": x[:3], "target": y[:3], **params}
    original = factory(graph).run(None, feeds)
    original[2][0, 0] = np.float32(2**-149)
    inspector = GraphInspector(graph, session_factory=factory)
    executor = inspector.executor

    class Subnormal:
        def run(self, *args):
            values = executor.run(*args)
            values[2][0, 0] = np.float32(2**-149)
            return values

    inspector.executor = Subnormal()
    report = inspector.check(feeds, original)
    assert report["original_outputs_equal"]
    assert all(row["violations"] == 0 for row in report["outputs"].values())
    assert not report["normal_or_zero_operands"]
    assert not report["all_pass"]


def test_fp64_reference_dot_error_is_included_before_fp32_padding():
    a = np.array([[2**40, np.nextafter(1.0, np.inf), -(2**40)]])
    b = np.ones((3, 1))
    exact = sum(fraction(x) * fraction(y) for x, y in zip(a[0], b[:, 0], strict=True))
    result = bounds.exact_matmul(bounds.Interval.point(a), bounds.Interval.point(b))
    assert fraction(result.lower[0, 0]) <= exact <= fraction(result.upper[0, 0])
    assert result.lower[0, 0] < result.upper[0, 0]


def test_mean_propagates_input_uncertainty_and_rounding():
    value = bounds.Interval(np.array([-2.0, 1.0, 3.0]), np.array([-1.0, 2.0, 4.0]))
    result = bounds.mean(value)
    assert fraction(result.lower) <= Fraction(2, 3)
    assert fraction(result.upper) >= Fraction(5, 3)
    with pytest.raises(ValueError, match="overflow"):
        bounds.fp32(bounds.Interval.point(2.0**128))


def test_mean_rejects_intermediate_overflow_even_for_a_finite_real_mean():
    with pytest.raises(ValueError, match="overflow"):
        bounds.mean(bounds.Interval.point(np.array([2.0**127, 2.0**127])))


def test_direct_bf16_rounding_handles_signed_midpoints_across_exponents():
    samples = []
    for exponent in (-134, -127, -80, -1, 0, 60, 120):
        midpoint = np.ldexp(1 + 2**-8, exponent)
        for sign in (-1, 1):
            value = sign * midpoint
            samples.extend([np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)])
    values = np.array(samples)
    actual = bounds.bf16(bounds.Interval.point(values)).lower
    expected = []
    for value in values:
        exact = fraction(value)
        expected.append(_round_endpoint(exact.numerator << 266, exact.denominator))
    np.testing.assert_array_equal(actual, expected)
