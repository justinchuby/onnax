# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Portable attention VJP functions preserve the original derivative graph."""

import json
import os

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import CheckerPass
from test_fused_autodiff import _model_for

import onnax


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("d", [8, 16, 40, 80, 160, 192, 200, 248, 256])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("mask_type", [None, "bool", "float"])
@pytest.mark.parametrize("selected", [["q"], ["k", "v"], ["q", "k", "v"]])
def test_portable_function(d, rank, mask_type, selected, recompute, causal):
    rng = np.random.default_rng(42)

    def shape(h, s):
        return (2, h, s, d) if rank == 4 else (2, s, h * d)

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
    attrs = {"is_causal": int(causal)}
    if rank == 3:
        attrs.update(q_num_heads=4, kv_num_heads=2)
    source = _model_for("Attention", feeds, attrs)
    derivative = onnax.vjp(source, inputs=selected)
    before = ir.to_proto(derivative).SerializeToString()
    optimized = onnax.FuseAttentionVJPPass(recompute=recompute)(derivative).model
    CheckerPass(full_check=True)(optimized)
    assert len(optimized.functions) == 1
    assert next(iter(optimized.functions))[0] == "pkg.onnax"
    assert len(optimized.graph) < len(derivative.graph)
    uses_recompute = recompute and 32 <= d <= 256
    operation = "AttentionVJPRecompute" if uses_recompute else "AttentionVJP"
    calls = [n for n in optimized.graph if n.op_type.startswith(f"{operation}_")]
    assert len(calls) == 1
    assert len(calls[0].inputs) == (5 if uses_recompute else 6)
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
        ir.to_proto(onnax.FuseAttentionVJPPass()(optimized).model).SerializeToString()
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
    feeds = {name: np.ones((1, 1, 2, 7), np.float32) for name in ("q", "k", "v")}
    derivative = onnax.vjp(_model_for("Attention", feeds))
    result = onnax.FuseAttentionVJPPass()(derivative).model
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


def test_untyped_internal_pullback_seed_keeps_native_boundary():
    rng = np.random.default_rng(9)
    feeds = {
        name: rng.normal(0, 0.2, (1, 2, 3, 256)).astype(np.float32) for name in ("q", "k", "v")
    }
    source = _model_for("Attention", feeds)
    output = source.graph.outputs[0]
    tape = ir.tape.Tape()
    gain = tape.initializer(ir.tensor(np.float32(2), name="gain"))
    scaled = tape.op("Mul", [output, gain])
    scaled.name = "scaled"
    scaled.dtype = ir.DataType.FLOAT
    scaled.shape = ir.Shape([1, 2, 3, 256])
    source.graph.extend(tape.nodes)
    source.graph.register_initializer(gain)
    source.graph.outputs.clear()
    source.graph.outputs.append(scaled)
    derivative = onnax.vjp(source, inputs=["q", "k", "v"])
    descriptor = next(
        json.loads(node.metadata_props["onnax.attention_vjp.interface"])
        for node in derivative.graph
        if "onnax.attention_vjp.interface" in node.metadata_props
    )
    value = ir.convenience.create_value_mapping(derivative.graph)[descriptor["inputs"][4]]
    assert value.producer() is not None
    value.type = value.shape = None
    before = ir.to_proto(derivative).SerializeToString()
    fused = onnax.FuseAttentionVJPPass()(derivative).model
    call = next(node for node in fused.graph if node.domain == "pkg.onnax")
    assert call.inputs[4].dtype == ir.DataType.FLOAT
    assert list(call.inputs[4].shape) == [1, 2, 3, 256]
    assert ir.to_proto(derivative).SerializeToString() == before
    feeds[derivative.graph.inputs[-1].name] = rng.normal(size=(1, 2, 3, 256)).astype(
        np.float32
    )
    for actual, expected in zip(
        runtime_session(fused).run(None, feeds),
        runtime_session(derivative).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_conflicting_seed_metadata_does_not_add_a_partial_function():
    feeds = {name: np.ones((1, 2, 3, 256), np.float32) for name in ("q", "k", "v")}
    derivative = onnax.vjp(_model_for("Attention", feeds))
    derivative.graph.inputs[-1].shape = ir.Shape([1, 2, 3, 512])
    before = ir.to_proto(derivative).SerializeToString()
    result = onnax.FuseAttentionVJPPass()(derivative).model
    assert not result.functions
    assert ir.to_proto(result).SerializeToString() == before


def test_attention_pullback_uses_row_sized_pivot_storage():
    feeds = {name: np.ones((1, 2, 3, 16), np.float32) for name in ("q", "k", "v")}
    derivative = onnax.vjp(_model_for("Attention", feeds), inputs=["q", "k", "v"])
    nodes = list(derivative.graph)
    assert not any(node.op_type == "ScatterElements" for node in nodes)
    scatters = [node for node in nodes if node.op_type == "ScatterND"]
    assert len(scatters) == 3
    # The two zero replacements share a vector with one entry per score row.
    zeros = scatters[0].inputs[2]
    assert zeros is scatters[1].inputs[2]
    zero_node = zeros.producer()
    assert zero_node.op_type == "ConstantOfShape"
    shape_node = zero_node.inputs[0].producer()
    assert shape_node.op_type == "Shape"
    flat_indices = shape_node.inputs[0]
    assert flat_indices.producer().op_type == "Add"
    coordinates = scatters[0].inputs[1]
    assert coordinates.producer().op_type == "Unsqueeze"
    assert coordinates.producer().inputs[0] is flat_indices
    CheckerPass(full_check=True)(derivative)


@pytest.mark.parametrize("d", [40, 80, 160])
def test_optimizer_preserves_unet_native_function_boundary(d):
    from onnxscript import optimizer

    rng = np.random.default_rng(7)
    feeds = {
        name: rng.normal(size=(1, 2, 3, d)).astype(np.float32) for name in ("q", "k", "v")
    }
    derivative = onnax.vjp(_model_for("Attention", feeds), inputs=["q", "k", "v"])
    optimized = optimizer.optimize(
        onnax.FuseAttentionVJPPass()(derivative).model, inline=False
    )
    assert len(optimized.functions) == 1
    assert sum(node.domain == "pkg.onnax" for node in optimized.graph) == 1
    CheckerPass(full_check=True)(optimized)
    feeds[derivative.graph.inputs[-1].name] = rng.normal(size=(1, 2, 3, d)).astype(np.float32)
    for actual, expected in zip(
        runtime_session(optimized).run(None, feeds),
        runtime_session(derivative).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def _small_derivative():
    feeds = {name: np.ones((1, 2, 3, 40), np.float32) for name in ("q", "k", "v")}
    return onnax.vjp(_model_for("Attention", feeds))


@pytest.mark.parametrize("recompute", [False, True])
def test_public_pass_composition_and_example_compatibility(recompute):
    from examples.huggingface.attention_vjp import fuse_attention_vjp

    model = _small_derivative()
    before = ir.to_proto(model).SerializeToString()
    result = ir.passes.PassManager([onnax.FuseAttentionVJPPass(recompute=recompute)])(model)
    assert result.modified
    assert not onnax.FuseAttentionVJPPass(recompute=recompute)(result.model).modified
    assert (
        ir.to_proto(result.model).SerializeToString()
        == ir.to_proto(fuse_attention_vjp(model, recompute=recompute)).SerializeToString()
    )
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize("recompute", [False, True])
def test_edited_body_keeps_its_original_computation(recompute):
    model = _small_derivative()
    node = next(
        node
        for node in model.graph
        if node.op_type == "Mul" and "onnax.attention_vjp.region" in node.metadata_props
    )
    node.op_type = "Add"
    before = ir.to_proto(model).SerializeToString()
    result = onnax.FuseAttentionVJPPass(recompute=recompute)(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize(
    "change", ["json", "inputs", "heads", "mask", "outputs", "attributes", "causal"]
)
def test_stale_or_malformed_descriptor_keeps_the_region(change):
    model = _small_derivative()
    node = next(
        node for node in model.graph if "onnax.attention_vjp.interface" in node.metadata_props
    )
    key = "onnax.attention_vjp.interface"
    spec = json.loads(node.metadata_props[key])
    if change == "json":
        node.metadata_props[key] = "not JSON"
    else:
        if change == "inputs":
            spec["inputs"] = spec["inputs"][:4]
        elif change == "heads":
            spec["hq"] = 7
        elif change == "mask":
            spec["mask"] = "missing-mask"
        elif change == "outputs":
            spec["outputs"] = {"3": "q"}
        elif change == "causal":
            spec["attributes"]["is_causal"] = 1
        else:
            spec["attributes"]["is_causal"] = 1.5
        node.metadata_props[key] = json.dumps(spec)
    before = ir.to_proto(model).SerializeToString()
    result = onnax.FuseAttentionVJPPass()(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before


@pytest.mark.parametrize("recompute", [False, True])
def test_incompatible_existing_function_keeps_source(recompute):
    model = _small_derivative()
    function = next(
        iter(onnax.FuseAttentionVJPPass(recompute=recompute)(model).model.functions.values())
    )
    next(node for node in function.graph if node.op_type == "Mul").op_type = "Add"
    model.functions[function.identifier()] = function
    before = ir.to_proto(model).SerializeToString()
    with pytest.raises(ValueError, match=r"incompatible pkg\.onnax attention VJP"):
        onnax.FuseAttentionVJPPass(recompute=recompute)(model)
    assert ir.to_proto(model).SerializeToString() == before


def test_recompute_selection_is_boolean():
    with pytest.raises(TypeError, match="must be a bool"):
        onnax.FuseAttentionVJPPass(recompute="yes")


def test_shadowed_standard_operation_keeps_portable_region():
    model = _small_derivative()
    a, b = [ir.val(name, dtype=ir.DataType.FLOAT, shape=[1]) for name in ("a", "b")]
    node = ir.node("Add", [a, b], num_outputs=1)
    function = ir.Function(
        "",
        "Mul",
        graph=ir.Graph([a, b], list(node.outputs), nodes=[node], opset_imports={"": 23}),
        attributes=[],
    )
    model.functions[function.identifier()] = function
    before = ir.to_proto(model).SerializeToString()
    result = onnax.FuseAttentionVJPPass()(model)
    assert not result.modified
    assert ir.to_proto(result.model).SerializeToString() == before


def test_symbolic_key_width_preserves_runtime_contract():
    model = _small_derivative()
    model.graph.inputs[1].shape = ir.Shape([1, 2, 3, "key_width"])
    result = onnax.FuseAttentionVJPPass(recompute=True)(model)
    assert result.modified
    feeds = {value.name: np.ones((1, 2, 3, 40), np.float32) for value in model.graph.inputs}
    for actual, expected in zip(
        runtime_session(result.model).run(None, feeds),
        runtime_session(model).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("provider", ["cpu", "mlx"])
@pytest.mark.parametrize("recompute", [False, True])
def test_connected_attention_regions_share_replaced_boundaries(recompute, provider):
    execute = runtime_session
    if provider == "mlx":
        if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
            pytest.skip("Native MLX opt-in")
        from examples.lora.mlx_runtime import register_library, session

        if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
            register_library(path)
        execute = session
    shape = [1, 2, 3, 40]
    inputs = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name in ("q", "k", "v", "next_k", "next_v")
    ]
    first = ir.val("first", dtype=ir.DataType.FLOAT, shape=shape)
    output = ir.val("output", dtype=ir.DataType.FLOAT, shape=shape)
    nodes = [
        ir.node("Attention", inputs[:3], outputs=[first]),
        ir.node("Attention", [first, *inputs[3:]], outputs=[output]),
    ]
    source = ir.Model(
        ir.Graph(
            inputs, [output], nodes=nodes, opset_imports={"": 23}, name="connected_attention"
        ),
        ir_version=10,
    )
    derivative = onnax.vjp(source)
    before = ir.to_proto(derivative).SerializeToString()
    result = onnax.FuseAttentionVJPPass(recompute=recompute)(derivative)
    assert result.modified
    assert sum(node.domain == "pkg.onnax" for node in result.model.graph) == 2
    CheckerPass(full_check=True)(result.model)
    rng = np.random.default_rng(71)
    feeds = {
        value.name: rng.normal(0, 0.2, shape).astype(np.float32)
        for value in derivative.graph.inputs
    }
    for actual, expected in zip(
        execute(result.model).run(None, feeds),
        runtime_session(derivative).run(None, feeds),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    assert ir.to_proto(derivative).SerializeToString() == before
