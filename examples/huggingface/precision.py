# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""FP64 Qwen3 oracle for separating graph semantics from FP32 rounding."""

import numpy as np
import onnx_ir as ir
import torch
import torch.nn.functional as functional


def double_graph(model):
    result = model.clone()
    for value in (
        *result.graph.inputs,
        *result.graph.initializers.values(),
        *(v for n in result.graph for v in n.outputs),
    ):
        if value.dtype == ir.DataType.FLOAT:
            value.type = ir.TensorType(ir.DataType.DOUBLE)
    for name, value in result.graph.initializers.items():
        if value.const_value.dtype == ir.DataType.FLOAT:
            value.const_value = ir.tensor(
                value.const_value.numpy().astype(np.float64), name=name
            )
    for node in result.graph:
        if node.op_type == "Cast" and node.attributes.get_int("to") == ir.DataType.FLOAT:
            node.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.DOUBLE))
        for name, attribute in list(node.attributes.items()):
            if (
                attribute.type == ir.AttributeType.TENSOR
                and attribute.as_tensor().dtype == ir.DataType.FLOAT
            ):
                node.attributes[name] = ir.AttrTensor(
                    name, ir.tensor(attribute.as_tensor().numpy().astype(np.float64))
                )
    return result


def qwen64(reference, model, arrays, batch):
    """Independent functional transformer; frozen matrices accumulate in FP64.

    Use the ONNX graph's stored RoPE tables, epsilon and attention scale to
    compare the same rounded constants while evaluating arithmetic in FP64.
    """
    params = {
        n: torch.tensor(v, dtype=torch.float64, requires_grad=True) for n, v in arrays.items()
    }
    cfg = reference.config
    table = model.graph.initializers
    cos_cache = torch.from_numpy(
        table["model.rotary_emb.cos_cache"].const_value.numpy().astype(np.float64)
    )
    sin_cache = torch.from_numpy(
        table["model.rotary_emb.sin_cache"].const_value.numpy().astype(np.float64)
    )
    ids, mask = batch["input_ids"], batch["attention_mask"]
    positions = batch.get("position_ids", (mask.cumsum(-1) - 1).clamp(min=0))
    cos, sin = cos_cache[positions].unsqueeze(1), sin_cache[positions].unsqueeze(1)
    cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    eps = float(np.float32(cfg.rms_norm_eps))
    scale = float(np.float32(cfg.head_dim**-0.5))

    def rms(x, layer):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * layer.weight.double()

    def linear(x, layer, name):
        base = layer.base if hasattr(layer, "base") else layer
        y = functional.linear(x, base.weight.double())
        if name + ".lora.A" in params:
            y = y + 0.25 * ((x @ params[name + ".lora.A"]) @ params[name + ".lora.B"])
        return y

    def rotate(x):
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat([-b, a], -1) * sin

    hidden = functional.embedding(ids, reference.model.embed_tokens.weight.double())
    for index, layer in enumerate(reference.model.layers):
        x = rms(hidden, layer.input_layernorm)
        prefix = f"model.layers.{index}.self_attn"
        attn = layer.self_attn
        q = (
            linear(x, attn.q_proj, prefix + ".q_proj")
            .reshape(*ids.shape, cfg.num_attention_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        k = (
            linear(x, attn.k_proj, prefix + ".k_proj")
            .reshape(*ids.shape, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        v = (
            linear(x, attn.v_proj, prefix + ".v_proj")
            .reshape(*ids.shape, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        q, k = rotate(rms(q, attn.q_norm)), rotate(rms(k, attn.k_norm))
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale
        causal = torch.ones(ids.shape[-1], ids.shape[-1], dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float(np.float32(-3e38)))
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), float(np.float32(-3e38)))
        context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(*ids.shape, -1)
        hidden = hidden + linear(context, attn.o_proj, prefix + ".o_proj")
        x = rms(hidden, layer.post_attention_layernorm)
        gate = linear(x, layer.mlp.gate_proj, "")
        up = linear(x, layer.mlp.up_proj, "")
        hidden = hidden + linear(functional.silu(gate) * up, layer.mlp.down_proj, "")
    hidden = rms(hidden, reference.model.norm)
    return functional.linear(hidden, reference.lm_head.weight.double()), params


def audit64(model, reference, batch, states):
    """Compare every logit and adapter derivative in FP64 at fixed states."""
    import tempfile

    import onnax
    from examples.huggingface.bridge import graph_session
    from examples.huggingface.optimization import optimize_training_graph

    graph = double_graph(model.forward_graph)
    names = list(model.onnx.names)
    gradient = optimize_training_graph(onnax.vjp(graph, inputs=names, outputs=["logits"]))
    with tempfile.TemporaryDirectory(prefix="onnax-fp64-") as directory:
        forward = graph_session(graph, directory, "forward")
        backward = graph_session(gradient, directory, "backward")
        return _audit64_sessions(
            model, reference, batch, states, graph, gradient, forward, backward
        )


def _audit64_sessions(model, reference, batch, states, graph, gradient, forward, backward):
    from examples.huggingface.sft import token_loss

    names = list(model.onnx.names)
    batch = dict(batch)
    if "position_ids" not in batch:
        batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    reports = []
    for arrays in states:
        inputs = {n: v.numpy() for n, v in batch.items() if n != "labels"}
        feeds = {**inputs, **{n: v.astype(np.float64) for n, v in arrays.items()}}
        prediction = forward.run(["logits"], feeds)[0]
        actual = torch.tensor(prediction, requires_grad=True)
        expected, params = qwen64(reference, model.forward_graph, arrays, batch)
        np.testing.assert_allclose(prediction, expected.detach(), rtol=1e-4, atol=1e-6)
        loss = token_loss(actual, batch["labels"])
        expected_loss = token_loss(expected, batch["labels"])
        np.testing.assert_allclose(loss.detach(), expected_loss.detach(), rtol=1e-4, atol=1e-6)
        loss.backward()
        expected_loss.backward()
        feeds[gradient.graph.inputs[-1].name] = actual.grad.numpy()
        values = backward.run(
            [v.name for v in gradient.graph.outputs[len(graph.graph.outputs) :]], feeds
        )
        errors = []
        for name, value in zip(names, values, strict=True):
            np.testing.assert_allclose(
                value, params[name].grad, rtol=1e-4, atol=1e-6, err_msg=name
            )
            errors.append(float(np.max(np.abs(value - params[name].grad.numpy()))))
        reports.append(
            {
                "logits_max_abs": float(
                    np.max(np.abs(prediction - expected.detach().numpy()))
                ),
                "gradients_max_abs": max(errors),
                "loss": float(loss.detach()),
                "strict_check_passed": True,
            }
        )
    return reports
