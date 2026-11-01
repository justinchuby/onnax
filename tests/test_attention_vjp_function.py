# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable attention VJP functions preserve the original derivative graph."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import CheckerPass
from test_fused_autodiff import _model_for

import onnax
from examples.huggingface.attention_vjp import fuse_attention_vjp


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("mask_type", [None, "bool", "float"])
@pytest.mark.parametrize("selected", [["q"], ["k", "v"], ["q", "k", "v"]])
def test_portable_function(rank, mask_type, selected):
    rng = np.random.default_rng(42)

    def shape(h, s):
        return (2, h, s, 16) if rank == 4 else (2, s, h * 16)

    feeds = {
        name: rng.normal(0, 0.2, shape(h, s)).astype(np.float32)
        for name, h, s in [("q", 4, 5), ("k", 2, 7), ("v", 2, 7)]
    }
    if mask_type is not None:
        allowed = rng.random((2, 1, 5, 7)) > 0.2
        allowed[0, :, 0] = False
        feeds["mask"] = (
            allowed
            if mask_type == "bool"
            else np.where(allowed, 0, -np.inf).astype(np.float32)
        )
    attrs = {"is_causal": 1}
    if rank == 3:
        attrs.update(q_num_heads=4, kv_num_heads=2)
    source = _model_for("Attention", feeds, attrs)
    derivative = onnax.vjp(source, inputs=selected)
    before = ir.to_proto(derivative).SerializeToString()
    optimized = fuse_attention_vjp(derivative)
    CheckerPass(full_check=True)(optimized)
    assert len(optimized.functions) == 1
    assert len(optimized.graph) < len(derivative.graph)
    assert any(n.op_type.startswith("AttentionVJP_v1_") for n in optimized.graph)
    assert ir.to_proto(derivative).SerializeToString() == before
    seed = derivative.graph.inputs[-1]
    output_shape = shape(4, 5)
    feeds[seed.name] = rng.normal(size=output_shape).astype(np.float32)
    # ORT CPU expands the function body; it supplies no native AttentionVJP kernel.
    expected = runtime_session(derivative).run(None, feeds)
    for actual, target in zip(
        runtime_session(optimized).run(None, feeds), expected, strict=True
    ):
        np.testing.assert_allclose(actual, target, rtol=1e-4, atol=1e-6)
    assert (
        ir.to_proto(fuse_attention_vjp(optimized)).SerializeToString()
        == ir.to_proto(optimized).SerializeToString()
    )


@pytest.mark.parametrize("boolean", [False, True])
def test_singleton_key_mask_gradient(boolean):
    rng = np.random.default_rng(4)
    feeds = {
        name: rng.normal(size=(1, 2, 3, 16)).astype(np.float32) for name in ("q", "k", "v")
    }
    feeds["mask"] = (
        np.ones((1, 1, 3, 1), bool) if boolean else np.zeros((1, 1, 3, 1), np.float32)
    )
    source = _model_for("Attention", feeds)
    derivative = onnax.vjp(source, inputs=["q", "k", "v"])
    plain = _model_for("Attention", {n: a for n, a in feeds.items() if n != "mask"})
    plain = onnax.vjp(plain, inputs=["q", "k", "v"])
    feeds[derivative.graph.inputs[-1].name] = rng.normal(size=(1, 2, 3, 16)).astype(np.float32)
    # Compare derivative-only execution with the unmasked broadcast equivalent.
    from onnx_ir.passes.common import RemoveUnusedNodesPass

    for result in (derivative, plain):
        gradients = list(result.graph.outputs)[1:]
        result.graph.outputs.clear()
        result.graph.outputs.extend(gradients)
        RemoveUnusedNodesPass()(result)
    unmasked = {n: a for n, a in feeds.items() if n != "mask"}
    for got, want in zip(
        runtime_session(derivative).run(None, feeds),
        runtime_session(plain).run(None, unmasked),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)


def test_unsupported_dimension_stays_portable():
    feeds = {name: np.ones((1, 1, 2, 8), np.float32) for name in ("q", "k", "v")}
    derivative = onnax.vjp(_model_for("Attention", feeds))
    result = fuse_attention_vjp(derivative)
    assert not result.functions
    assert (
        ir.to_proto(result).SerializeToString() == ir.to_proto(derivative).SerializeToString()
    )


@pytest.mark.parametrize("transform", [onnax.jvp, onnax.vjp])
def test_singleton_additive_mask_has_zero_product(transform):
    from onnx_ir.passes.common import RemoveUnusedNodesPass

    rng = np.random.default_rng(17)
    feeds = {
        name: rng.normal(size=(1, 2, 3, 16)).astype(np.float32) for name in ("q", "k", "v")
    }
    feeds["mask"] = rng.normal(size=(1, 1, 3, 1)).astype(np.float32)
    product = transform(_model_for("Attention", feeds), inputs=["mask"])
    seed = product.graph.inputs[-1]
    feeds[seed.name] = rng.normal(
        size=(1, 1, 3, 1) if transform is onnax.jvp else (1, 2, 3, 16)
    ).astype(np.float32)
    outputs = list(product.graph.outputs)[1:]
    product.graph.outputs.clear()
    product.graph.outputs.extend(outputs)
    RemoveUnusedNodesPass()(product)
    actual = runtime_session(product).run(None, feeds)[0]
    np.testing.assert_allclose(actual, np.zeros_like(actual), rtol=0, atol=1e-6)
