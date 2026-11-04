# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Gated-delta products through the actual Mobius portable function."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import runtime_session
from onnx_ir.passes.common import CheckerPass
from test_fused_autodiff import _check

import onnax
from examples.huggingface.linear_attention_reference import gated_delta, pullback

pytest.importorskip("mobius")
from mobius.functions.linear_attention import linear_attention


def model_and_arrays(query_heads=1, value_heads=2, length=7, *, keywise_decay=True):
    rng = np.random.default_rng(71)
    shapes = [(2, length, query_heads * 3)] * 2 + [
        (2, length, value_heads * 5),
        (2, value_heads, 3, 5),
        (2, length, value_heads * (3 if keywise_decay else 1)),
        (2, length, value_heads),
    ]
    arrays = {
        name: rng.normal(0, 0.2, shape).astype(np.float32)
        for name, shape in zip(("q", "k", "v", "state", "decay", "beta"), shapes, strict=True)
    }
    arrays["decay"] = -np.abs(arrays["decay"])
    inputs = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name, shape in zip(arrays, shapes, strict=True)
    ]
    for index, value in enumerate(inputs):
        value.shape = ir.Shape(["batch", *shapes[index][1:]])
        if index != 3:
            value.shape = ir.Shape(["batch", "time", shapes[index][2]])
    function = linear_attention(q_num_heads=query_heads, kv_num_heads=value_heads, scale=0.7)
    tape = ir.tape.Tape()
    outputs = tape.op_multi_out(
        "LinearAttention",
        inputs,
        {
            "q_num_heads": query_heads,
            "kv_num_heads": value_heads,
            "scale": 0.7,
            "update_rule": "gated_delta",
        },
        domain="com.microsoft",
        num_outputs=2,
    )
    for index, output in enumerate(outputs):
        output.name = f"output{index}"
        output.dtype = ir.DataType.FLOAT
        output.shape = ir.Shape(inputs[2 if index == 0 else 3].shape)
    return ir.Model(
        ir.Graph(
            inputs,
            outputs,
            nodes=tape.nodes,
            opset_imports={"": 24, "com.microsoft": 1},
            name="linear_attention",
        ),
        ir_version=10,
        functions=[function],
    ), arrays


def reference(tensors, query_heads, value_heads):
    import torch

    q, k, v, state, decay, beta = tensors
    batch, length, _ = q.shape
    q = q.reshape(batch, length, query_heads, 3).repeat_interleave(
        value_heads // query_heads, 2
    )
    k = k.reshape(batch, length, query_heads, 3).repeat_interleave(
        value_heads // query_heads, 2
    )
    v = v.reshape(batch, length, value_heads, 5)
    decay = decay.reshape(batch, length, value_heads, -1)
    outputs = []
    for index in range(length):
        faded = torch.exp(decay[:, index])[..., None] * state
        residual = v[:, index] - (k[:, index, :, None, :] @ faded).squeeze(-2)
        state = (
            faded
            + beta[:, index, :, None, None] * k[:, index, :, :, None] * residual[..., None, :]
        )
        outputs.append((q[:, index, :, None, :] @ state).squeeze(-2) * 0.7)
    return torch.stack(outputs, dim=1).reshape(tensors[2].shape), state


@pytest.mark.parametrize("query_heads,value_heads", [(1, 1), (1, 3), (2, 4)])
@pytest.mark.parametrize("length", [1, 7])
@pytest.mark.parametrize("names", [None, ["state"], ["decay", "beta"], ["q"]])
def test_directional_recurrence(query_heads, value_heads, length, names):
    torch = pytest.importorskip("torch")
    source, arrays = model_and_arrays(query_heads, value_heads, length)
    CheckerPass(full_check=True)(source)
    before = ir.to_proto(source).SerializeToString()
    names = names or list(arrays)
    product = onnax.jvp(source, inputs=names)
    CheckerPass(full_check=True)(product)
    rng = np.random.default_rng(19)
    directions = {
        name: rng.normal(size=arrays[name].shape).astype(np.float32) for name in names
    }
    leaves = [torch.tensor(x) for x in arrays.values()]
    tangent = [
        torch.tensor(directions[name]) if name in directions else torch.zeros_like(value)
        for name, value in zip(arrays, leaves, strict=True)
    ]
    expected, expected_jvp = torch.autograd.functional.jvp(
        lambda *args: reference(args, query_heads, value_heads), tuple(leaves), tuple(tangent)
    )
    feeds = dict(arrays)
    feeds.update(
        (value.name, directions[name])
        for value, name in zip(list(product.graph.inputs)[6:], names, strict=True)
    )
    actual = runtime_session(product).run(None, feeds)
    oracle = gated_delta(
        *arrays.values(), q_num_heads=query_heads, kv_num_heads=value_heads, scale=0.7
    )
    for result, ref in zip(actual, (*expected, *expected_jvp), strict=True):
        np.testing.assert_allclose(result, ref.numpy(), rtol=1e-4, atol=1e-6)
    for result, ref in zip(actual[:2], oracle, strict=True):
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-6)
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("query_heads,value_heads", [(1, 1), (1, 3), (2, 4)])
@pytest.mark.parametrize("length", [1, 7, 33, 65])
@pytest.mark.parametrize("names", [None, ["state"], ["decay", "beta"], ["q"]])
@pytest.mark.parametrize("keywise_decay", [False, True])
def test_chunk_recomputed_pullback(query_heads, value_heads, length, names, keywise_decay):
    source, arrays = model_and_arrays(
        query_heads, value_heads, length, keywise_decay=keywise_decay
    )
    _check(
        source,
        arrays,
        lambda tensors: reference(list(tensors.values()), query_heads, value_heads),
        names=names,
    )


@pytest.mark.parametrize("cotangent_mode", ["output", "state", "both"])
def test_dynamic_sequence_and_independent_output_cotangents(cotangent_mode):
    source, _ = model_and_arrays()
    product = onnax.vjp(source)
    session = runtime_session(product)
    rng = np.random.default_rng(31)
    for length in (1, 32, 33, 65):
        _, arrays = model_and_arrays(length=length)
        # Reuse the same session with changing sequence length and batch size.
        arrays = {name: value[:1] for name, value in arrays.items()}
        gy = rng.normal(size=arrays["v"].shape).astype(np.float32)
        gs = rng.normal(size=arrays["state"].shape).astype(np.float32)
        if cotangent_mode == "output":
            gs.fill(0)
        if cotangent_mode == "state":
            gy.fill(0)
        feeds = dict(arrays)
        feeds.update(
            (value.name, array)
            for value, array in zip(list(product.graph.inputs)[6:], (gy, gs), strict=True)
        )
        actual = session.run(None, feeds)[2:]
        expected, _ = pullback(
            *arrays.values(),
            gy,
            gs,
            q_num_heads=1,
            kv_num_heads=2,
            scale=0.7,
            chunk_size=32,
        )
        for result, gradient in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, gradient, rtol=1e-4, atol=1e-6)


def test_checkpoint_plan_minimizes_retained_state_tiles():
    from onnax._autodiff._context import _Context
    from onnax._autodiff._linear_attention import _checkpoint_chunk

    time = ir.val("time", dtype=ir.DataType.INT64, shape=["cases"])
    graph = ir.Graph([time], [], nodes=[], opset_imports={"": 18}, name="checkpoint_plan")
    chunk = _checkpoint_chunk(_Context(graph), time)
    chunk.name, chunk.dtype, chunk.shape = "chunk", ir.DataType.INT64, ir.Shape(["cases"])
    graph.outputs.append(chunk)
    model = ir.Model(graph, ir_version=10)
    CheckerPass(full_check=True)(model)
    lengths = np.array([1, 32, 65, 1024, 1025, 2048, 2081, 4032, 4096, 8192, 32768], np.int64)
    chunks = runtime_session(model).run(None, {"time": lengths})[0]
    np.testing.assert_array_equal(chunks, [32, 32, 32, 32, 32, 32, 64, 64, 64, 96, 192])

    def cost(length, chunk):
        return (int(length) + chunk - 1) // chunk + chunk

    for length, actual in zip(lengths, chunks, strict=True):
        assert cost(length, int(actual)) == min(cost(length, c) for c in range(32, 1025, 32))
        assert cost(length, int(actual)) <= cost(length, 32)


def test_adaptive_replay_reuses_session_across_tile_sizes():
    source, _ = model_and_arrays()
    product = onnax.vjp(source)
    CheckerPass(full_check=True)(product)
    session = runtime_session(product)
    rng = np.random.default_rng(195)
    for length in (33, 2081, 8192, 7):
        _, arrays = model_and_arrays(length=length)
        arrays = {name: value[:1] for name, value in arrays.items()}
        gy = rng.normal(size=arrays["v"].shape).astype(np.float32)
        gs = rng.normal(size=arrays["state"].shape).astype(np.float32)
        feeds = dict(arrays)
        feeds.update(
            (value.name, array)
            for value, array in zip(list(product.graph.inputs)[6:], (gy, gs), strict=True)
        )
        actual = session.run(None, feeds)[2:]
        expected, _ = pullback(
            *arrays.values(), gy, gs, q_num_heads=1, kv_num_heads=2, scale=0.7, chunk_size=32
        )
        for result, gradient in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, gradient, rtol=1e-4, atol=1e-6)
