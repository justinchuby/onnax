# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""GELU products for both ONNX approximation modes and their composition."""

import math

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from test_autodiff import _model, _products, _value
from test_higher_order import _hvp_check

from onnax import JvpPass, VjpPass, grad, jvp, vjp, vmap


def _gelu_model(approximation, *, opset=20, scalar=False, dtype=ir.DataType.DOUBLE):
    x = _value("x", ["n"], dtype=dtype)
    tape = ir.tape.Tape()
    attributes = {} if approximation is None else {"approximate": approximation}
    y = tape.op("Gelu", [x], attributes)
    if scalar:
        y = tape.op("ReduceSum", [y], {"keepdims": 0})
    y.dtype = dtype
    y.shape = ir.Shape([] if scalar else ["n"])
    return _model([x], tape, [y], opset=opset)


@pytest.mark.parametrize("approximation", [None, "none", "tanh"])
@pytest.mark.parametrize("opset", [20, 24, 28])
def test_gelu_products(approximation, opset):
    # Reference Erf rounds through float32; use a finite-difference step that
    # separates truncation error from that oracle's intermediate rounding.
    options = {"epsilon": 1e-3, "rtol": 4e-4, "atol": 3e-5} if approximation != "tanh" else {}
    _products(
        _gelu_model(approximation, opset=opset),
        {"x": np.array([-2.0, -0.5, 0.0, 0.4, 1.5])},
        **options,
    )


@pytest.mark.parametrize("approximation", [None, "none", "tanh"])
def test_gelu_hvp(approximation):
    x = _value("x", ["n", 2])
    tape = ir.tape.Tape()
    attributes = {} if approximation is None else {"approximate": approximation}
    y = tape.op("Gelu", [x], attributes)
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    options = {"epsilon": 2e-3, "rtol": 2e-3, "atol": 2e-4} if approximation != "tanh" else {}
    _hvp_check(
        _model([x], tape, [loss], opset=20),
        {"x": np.array([[-1.2, -0.4], [0.0, 1.1]])},
        **options,
    )


@pytest.mark.parametrize("approximation", ["none", "tanh"])
def test_gelu_runtime_and_batched_gradients(approximation):
    model = _gelu_model(approximation, scalar=True, dtype=ir.DataType.FLOAT)
    values = np.array([-1e20, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 1e20], dtype=np.float32)
    x = values.astype(np.float64)
    if approximation == "none":
        density = np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        expected = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x])) + x * density
        curvature = (2 - x**2) * density
    else:
        scale, cubic = math.sqrt(2 / math.pi), 0.044715
        t = np.tanh(scale * (x + cubic * x**3))
        first, second = scale * (1 + 3 * cubic * x**2), scale * 6 * cubic * x
        expected = 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * first
        curvature = (1 - t**2) * (first + 0.5 * x * (second - 2 * t * first**2))

    forward, reverse = jvp(model), vjp(model)
    actual = runtime_session(forward).run(
        None, {"x": values, forward.graph.inputs[-1].name: np.ones_like(values)}
    )
    np.testing.assert_allclose(actual[-1], expected.sum(), rtol=2e-6, atol=1e-7)
    actual = runtime_session(reverse).run(
        None, {"x": values, reverse.graph.inputs[-1].name: np.array(1, dtype=np.float32)}
    )
    np.testing.assert_allclose(actual[-1], expected, rtol=2e-5, atol=1e-7)

    gradient = grad(model)
    product = ir.from_proto(ir.to_proto(jvp(gradient)))
    actual = runtime_session(product).run(
        None, {"x": values, product.graph.inputs[-1].name: np.ones_like(values)}
    )
    assert all(np.isfinite(output).all() for output in actual)
    np.testing.assert_allclose(actual[-1], curvature, rtol=3e-5, atol=1e-7)
    batched = vmap(gradient, in_axes=(0,))
    actual = runtime_session(batched).run(None, {"x": np.stack([values, values[::-1]])})[-1]
    np.testing.assert_allclose(
        actual, np.stack([expected, expected[::-1]]), rtol=2e-5, atol=1e-7
    )


@pytest.mark.parametrize("transform", [JvpPass, VjpPass])
def test_gelu_invalid_approximation_is_actionable(transform):
    with pytest.raises(
        ValueError, match=r"Gelu.*approximate='unsupported'.*expected 'none' or 'tanh'"
    ):
        transform()(_gelu_model("unsupported"))


@pytest.mark.parametrize("transform", [JvpPass, VjpPass])
def test_gelu_requires_opset_20(transform):
    with pytest.raises(NotImplementedError, match=r"Gelu.*20 through 28.*opset 19"):
        transform()(_gelu_model("none", opset=19))
