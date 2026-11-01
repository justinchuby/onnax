# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Fused primals and their products against independent PyTorch autograd."""

import numpy as np
import onnx_ir as ir
import pytest
from _testing import evaluator, runtime_session
from onnx_ir.passes.common import CheckerPass
from test_autodiff import _model

from onnax import jvp, vjp


def _model_for(op, arrays, attributes=None, *, outputs=1, symbolic=False, opset=25):
    inputs = [
        ir.val(
            name,
            dtype=ir.DataType.from_numpy(value.dtype),
            shape=(
                [
                    f"{name}_{i}" if i in (0, 2) or value.ndim == 2 else n
                    for i, n in enumerate(value.shape)
                ]
                if symbolic
                else value.shape
            ),
        )
        for name, value in arrays.items()
    ]
    tape = ir.tape.Tape()
    values = tape.op_multi_out(op, inputs, attributes or {}, num_outputs=outputs)
    return _model(inputs, tape, list(values), opset=opset)


def _check(model, feeds, reference, names=None, rtol=1e-4, atol=1e-6):
    torch = pytest.importorskip("torch")
    names = names or [name for name, value in feeds.items() if value.dtype == np.float32]
    tensors = {
        name: torch.tensor(value, requires_grad=name in names) for name, value in feeds.items()
    }
    outputs = reference(tensors)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    rng = np.random.default_rng(20261101)
    seeds = [rng.normal(size=value.shape).astype(np.float32) for value in outputs]
    directions = [rng.normal(size=feeds[name].shape).astype(np.float32) for name in names]
    expected = torch.autograd.grad(
        outputs, [tensors[name] for name in names], [torch.tensor(seed) for seed in seeds]
    )

    def function(*args):
        values = dict(tensors)
        values.update(zip(names, args))
        return reference(values)

    _, tangent = torch.autograd.functional.jvp(
        function,
        tuple(tensors[name] for name in names),
        tuple(torch.tensor(d) for d in directions),
    )
    if isinstance(tangent, torch.Tensor):
        tangent = (tangent,)
    before = ir.to_proto(model).SerializeToString()
    for mode, products, extras in [(vjp, expected, seeds), (jvp, tangent, directions)]:
        result = mode(model, inputs=names)
        CheckerPass(full_check=True)(result)
        assert (
            sum(node.op_type == next(iter(model.graph)).op_type for node in result.graph) >= 1
        )
        data = dict(feeds)
        data.update(
            (value.name, extra)
            for value, extra in zip(result.graph.inputs[len(feeds) :], extras)
        )
        executor = (
            evaluator(result)
            if next(iter(model.graph)).op_type == "Swish"
            or "left_window_size" in next(iter(model.graph)).attributes
            or ("mask" in feeds and feeds["mask"].shape[-1] < 5)
            else runtime_session(result)
        )
        actual = executor.run(None, data)
        for got, want in zip(actual[: len(outputs)], outputs):
            np.testing.assert_allclose(got, want.detach().numpy(), rtol=rtol, atol=atol)
        for got, want in zip(actual[len(outputs) :], products):
            np.testing.assert_allclose(got, want.detach().numpy(), rtol=rtol, atol=atol)
    assert ir.to_proto(model).SerializeToString() == before


@pytest.mark.parametrize(
    "op,stats",
    [("RMSNormalization", False), ("LayerNormalization", False), ("LayerNormalization", True)],
)
@pytest.mark.parametrize("axis", [-1, 1])
def test_normalization(op, axis, stats):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    shape = (2, 3, 4)
    arrays = {
        "x": rng.normal(size=shape).astype(np.float32),
        "scale": rng.normal(size=shape[axis:]).astype(np.float32),
    }
    if op == "LayerNormalization":
        arrays["bias"] = rng.normal(size=shape[axis:]).astype(np.float32)

    def reference(t):
        dims = tuple(range(axis % 3, 3))
        mean = t["x"].mean(dims, keepdim=True)
        x = t["x"] - mean if op == "LayerNormalization" else t["x"]
        inverse = torch.rsqrt((x * x).mean(dims, keepdim=True) + 1e-5)
        y = x * inverse * t["scale"]
        if "bias" in t:
            y = y + t["bias"]
        return (y, mean, inverse) if stats else y

    model = _model_for(op, arrays, {"axis": axis}, outputs=3 if stats else 1)
    _check(model, arrays, reference)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.7])
def test_swish(alpha):
    torch = pytest.importorskip("torch")
    arrays = {"x": np.array([-30.0, -2.0, -0.1, 0.0, 1.0, 30.0], np.float32)}
    _check(
        _model_for("Swish", arrays, {"alpha": alpha}),
        arrays,
        lambda t: t["x"] * torch.sigmoid(alpha * t["x"]),
    )


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("interleaved", [0, 1])
@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_rope(rank, interleaved, positions, partial):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    width = 4 if partial else 8
    arrays = {
        "x": rng.normal(size=(2, 3, 16) if rank == 3 else (2, 2, 3, 8)).astype(np.float32)
    }
    arrays.update(
        {
            name: rng.normal(size=(6, width // 2) if positions else (2, 3, width // 2)).astype(
                np.float32
            )
            for name in ("cos", "sin")
        }
    )
    if positions:
        arrays["position"] = np.array([[0, 2, 2], [4, 0, 1]], np.int64)
    attrs = {"interleaved": interleaved, "rotary_embedding_dim": width}
    if rank == 3:
        attrs["num_heads"] = 2

    def reference(t):
        x = t["x"].reshape(2, 3, 2, 8) if rank == 3 else t["x"].transpose(1, 2)
        c, s = t["cos"], t["sin"]
        if positions:
            c, s = c[t["position"]], s[t["position"]]
        c, s = c.unsqueeze(2), s.unsqueeze(2)
        a, b = (
            (x[..., :width:2], x[..., 1:width:2])
            if interleaved
            else (x[..., : width // 2], x[..., width // 2 : width])
        )
        real, imaginary = a * c - b * s, a * s + b * c
        rotated = (
            torch.stack((real, imaginary), -1).flatten(-2)
            if interleaved
            else torch.cat((real, imaginary), -1)
        )
        y = torch.cat((rotated, x[..., width:]), -1)
        return y.reshape(2, 3, 16) if rank == 3 else y.transpose(1, 2)

    _check(_model_for("RotaryEmbedding", arrays, attrs), arrays, reference)


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "float", "short"])
def test_attention(rank, kv_heads, mask_kind):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    q = rng.normal(size=(2, 4, 3, 4)).astype(np.float32)
    k = rng.normal(size=(2, kv_heads, 5, 4)).astype(np.float32)
    v = rng.normal(size=(2, kv_heads, 5, 6)).astype(np.float32)
    arrays = dict(zip(("q", "k", "v"), (q, k, v)))
    if rank == 3:
        arrays = {
            name: x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
            for name, x in arrays.items()
        }
    attrs = {"is_causal": 1, "softcap": 2.0}
    if rank == 3:
        attrs.update(q_num_heads=4, kv_num_heads=kv_heads)
    if mask_kind == "bool":
        mask = rng.random(size=(2, 1, 3, 5)) > 0.4
        mask[0, :, 0, :] = False
        arrays["mask"] = mask
    elif mask_kind in ("float", "short"):
        arrays["mask"] = rng.normal(size=(3, 3 if mask_kind == "short" else 5)).astype(
            np.float32
        )

    def reference(t):
        qs, ks, vs = t["q"], t["k"], t["v"]
        if rank == 3:
            qs = qs.reshape(2, 3, 4, 4).transpose(1, 2)
            ks = ks.reshape(2, 5, kv_heads, 4).transpose(1, 2)
            vs = vs.reshape(2, 5, kv_heads, 6).transpose(1, 2)
        ks, vs = ks.repeat_interleave(4 // kv_heads, 1), vs.repeat_interleave(4 // kv_heads, 1)
        scores = (qs @ ks.transpose(-1, -2)) * 0.5
        scores = 2 * torch.tanh(scores / 2)
        causal = torch.arange(5)[None, :] <= torch.arange(3)[:, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        if "mask" in t:
            mask = t["mask"]
            if mask.shape[-1] < 5:
                mask = torch.nn.functional.pad(
                    mask, (0, 5 - mask.shape[-1]), value=float("-inf")
                )
            scores = (
                scores.masked_fill(~mask, float("-inf"))
                if mask.dtype == torch.bool
                else scores + mask
            )
        empty = torch.isneginf(scores).all(-1, keepdim=True)
        probabilities = torch.softmax(torch.where(empty, 0, scores), -1)
        probabilities = torch.where(empty, 0, probabilities)
        y = probabilities @ vs
        return y.transpose(1, 2).reshape(2, 3, 24) if rank == 3 else y

    _check(_model_for("Attention", arrays, attrs), arrays, reference)


@pytest.mark.parametrize(
    "op", ["Attention", "RMSNormalization", "LayerNormalization", "RotaryEmbedding", "Swish"]
)
def test_portable_products_without_torch(op):
    """Keep finite-difference and duality coverage in the core dependency CI."""
    from test_autodiff import _products

    rng = np.random.default_rng(8)
    attrs = {}
    if op == "Attention":
        arrays = {
            "q": rng.normal(size=(1, 2, 2, 2)),
            "k": rng.normal(size=(1, 1, 3, 2)),
            "v": rng.normal(size=(1, 1, 3, 2)),
        }
        attrs = {"is_causal": 1}
    elif op == "RotaryEmbedding":
        arrays = {
            "x": rng.normal(size=(1, 2, 2, 4)),
            "cos": rng.normal(size=(1, 2, 2)),
            "sin": rng.normal(size=(1, 2, 2)),
        }
    elif op in ("RMSNormalization", "LayerNormalization"):
        arrays = {"x": rng.normal(size=(2, 3)), "scale": rng.normal(size=(3,))}
        arrays["x"] = np.array([[-2.0, 0.5, 2.0], [-1.0, 1.0, 3.0]])
        attrs = {}
    else:
        arrays = {"x": rng.normal(size=(4,))}
    arrays = {name: value.astype(np.float32) for name, value in arrays.items()}
    model = _model_for(op, arrays, attrs)
    _products(model, arrays, epsilon=1e-3, rtol=2e-3, atol=2e-4)
    from test_higher_order import _hvp_check

    tape = ir.tape.Tape(model.graph)
    y = model.graph.outputs[0]
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    loss.name = "loss"
    model.graph.outputs.clear()
    model.graph.outputs.append(loss)
    _hvp_check(
        model,
        arrays,
        epsilon=5e-3 if op == "LayerNormalization" else 1e-3,
        rtol=2e-3,
        atol=3e-4,
    )


@pytest.mark.parametrize(
    "op,minimum",
    [("Attention", 23), ("RMSNormalization", 23), ("RotaryEmbedding", 23), ("Swish", 24)],
)
def test_fused_opset_errors(op, minimum):
    arrays = {"x": np.ones((2, 3), np.float32)}
    model = _model_for(op, arrays, opset=minimum - 1)
    model.graph.outputs[0].dtype = ir.DataType.FLOAT
    model.graph.outputs[0].shape = ir.Shape([2, 3])
    for transform in (jvp, vjp):
        with pytest.raises(
            (NotImplementedError, ValueError), match=r"(opset|schema|registered)"
        ):
            transform(model)


@pytest.mark.parametrize("opset", [23, 24, 25, 28])
def test_attention_versions_and_window(opset):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    arrays = {
        name: rng.normal(size=(1, 2, 4, 2)).astype(np.float32) for name in ("q", "k", "v")
    }
    attrs = {"scale": 0.3}
    if opset >= 25:
        attrs.update(left_window_size=1, right_window_size=1)

    def reference(t):
        scores = (t["q"] @ t["k"].transpose(-1, -2)) * np.float32(0.3)
        if opset >= 25:
            positions = torch.arange(4)
            mask = (positions[:, None] - positions[None, :]).abs() <= 1
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, -1) @ t["v"]

    _check(_model_for("Attention", arrays, attrs, opset=opset), arrays, reference)


@pytest.mark.parametrize("selected", [["q"], ["v"], ["mask"]])
def test_attention_selected_inputs_and_dynamic_shapes(selected):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    arrays = {
        name: rng.normal(size=(2, 2, 3, 4)).astype(np.float32) for name in ("q", "k", "v")
    }
    arrays["mask"] = rng.normal(size=(3, 3)).astype(np.float32)
    model = _model_for("Attention", arrays, {"scale": 0.5}, symbolic=True)

    def reference(t):
        scores = (t["q"] @ t["k"].transpose(-1, -2)) * 0.5 + t["mask"]
        return torch.softmax(scores, -1) @ t["v"]

    _check(model, arrays, reference, names=selected)
    # Reuse the authored symbolic model with a different batch and sequence.
    arrays = {
        name: rng.normal(size=(1, 2, 2, 4)).astype(np.float32) for name in ("q", "k", "v")
    }
    arrays["mask"] = rng.normal(size=(2, 2)).astype(np.float32)
    _check(model, arrays, reference, names=selected)


def test_attention_auxiliary_outputs_require_explicit_support():
    arrays = {name: np.ones((1, 1, 2, 2), np.float32) for name in ("q", "k", "v")}
    model = _model_for("Attention", arrays, outputs=3)
    for transform in (jvp, vjp):
        with pytest.raises(NotImplementedError, match="cache-free training"):
            transform(model, outputs=["y0"])


def test_rotary_hvp_with_repeated_positions():
    from test_higher_order import _hvp_check

    rng = np.random.default_rng(14)
    arrays = {
        "x": rng.normal(size=(1, 1, 2, 4)).astype(np.float32),
        "cos": rng.normal(size=(3, 1)).astype(np.float32),
        "sin": rng.normal(size=(3, 1)).astype(np.float32),
        "position": np.array([[1, 1]], np.int64),
    }
    model = _model_for("RotaryEmbedding", arrays, {"rotary_embedding_dim": 2})
    tape = ir.tape.Tape(model.graph)
    y = model.graph.outputs[0]
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    loss.name = "loss"
    model.graph.outputs.clear()
    model.graph.outputs.append(loss)
    _hvp_check(model, arrays, inputs=["x", "cos", "sin"], epsilon=1e-3, rtol=2e-3, atol=3e-4)


def test_attention_hvp_with_additive_mask():
    from test_higher_order import _hvp_check

    rng = np.random.default_rng(15)
    arrays = {
        name: rng.normal(size=(1, 1, 2, 2)).astype(np.float32) for name in ("q", "k", "v")
    }
    arrays["mask"] = rng.normal(size=(2, 2)).astype(np.float32)
    model = _model_for("Attention", arrays)
    tape = ir.tape.Tape(model.graph)
    y = model.graph.outputs[0]
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    loss.name = "loss"
    model.graph.outputs.clear()
    model.graph.outputs.append(loss)
    _hvp_check(model, arrays, epsilon=1e-3, rtol=2e-3, atol=3e-4)


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap"])
def test_pad_jvp_boundaries(mode):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
    fill = ir.val("fill", dtype=ir.DataType.FLOAT, shape=[])
    tape = ir.tape.Tape()
    pads = tape.op("Constant", [], {"value": ir.tensor([1, 1], dtype=ir.DataType.INT64)})
    y = tape.op("Pad", [x, pads, fill], {"mode": mode})
    model = _model([x, fill], tape, [y])
    result = jvp(model)
    data = {
        "x": np.array([1, 2, 3], np.float32),
        "fill": np.array(7, np.float32),
        result.graph.inputs[-2].name: np.array([0.1, 0.2, 0.3], np.float32),
        result.graph.inputs[-1].name: np.array(0.4, np.float32),
    }
    actual = runtime_session(result).run(None, data)[-1]
    expected = np.pad(
        data[result.graph.inputs[-2].name],
        1,
        mode=mode,
        **({"constant_values": 0.4} if mode == "constant" else {}),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.7])
def test_swish_curvature_at_zero(alpha):
    from onnax import grad

    arrays = {"x": np.zeros(3, np.float32)}
    model = _model_for("Swish", arrays, {"alpha": alpha})
    tape = ir.tape.Tape(model.graph)
    loss = tape.op("ReduceSum", [model.graph.outputs[0]], {"keepdims": 0})
    loss.name = "loss"
    model.graph.outputs.clear()
    model.graph.outputs.append(loss)
    product = jvp(grad(model))
    actual = evaluator(product).run(
        None, {"x": arrays["x"], product.graph.inputs[-1].name: np.ones(3, np.float32)}
    )[-1]
    np.testing.assert_allclose(actual, np.full(3, alpha / 2, np.float32), rtol=1e-6, atol=0)


def test_attention_mixed_precision_inputs_report_supported_compute_dtype():
    arrays = {
        "q": np.ones((1, 1, 2, 2), np.float32),
        "k": np.ones((1, 1, 2, 2), np.float16),
        "v": np.ones((1, 1, 2, 2), np.float16),
    }
    model = _model_for("Attention", arrays)
    for transform in (jvp, vjp):
        with pytest.raises(NotImplementedError, match="matching FLOAT or DOUBLE"):
            transform(model, inputs=["q"])
