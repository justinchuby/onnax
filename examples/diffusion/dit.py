# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline Mobius DiT construction and example-local attention adapter injection."""

import numpy as np
import onnx_ir as ir
from mobius.models.dit import DiTConfig, DiTTransformer2DModel
from mobius.tasks import DenoisingTask
from onnx_ir.passes.common import ShapeInferencePass


def build():
    """Build a two-layer conditional latent denoiser without tracing or downloads."""
    config = DiTConfig(
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        cross_attention_dim=32,
        caption_channels=32,
        sample_size=8,
    )
    model = DenoisingTask().build(DiTTransformer2DModel(config), config)["model"]
    rng = np.random.default_rng(20261102)
    weights = {}
    for name, value in model.graph.initializers.items():
        if value.const_value is not None:
            continue
        shape = tuple(value.shape)
        if name.endswith(".bias"):
            array = np.zeros(shape, np.float32)
        elif ".norm." in name and name.endswith(".weight"):
            array = np.ones(shape, np.float32)
        else:
            array = rng.normal(0, 0.06, shape).astype(np.float32)
        value.const_value = ir.tensor(array, name=name)
        weights[name] = array.copy()
    ShapeInferencePass()(model)
    return config, model, weights


def inject(model, *, rank=4, alpha=4.0):
    """Adapt self/cross-attention q/v projections; keep source Linear names."""
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
        or not np.isfinite(alpha)
    ):
        raise ValueError("Expected a positive integer rank and finite alpha")
    result = model.clone()
    params, mapping = {}, []
    existing = set(ir.convenience.create_value_mapping(result.graph))
    for node in list(result.graph):
        if node.domain or node.op_type != "MatMul":
            continue
        transposed = node.inputs[1].producer()
        if transposed is None or transposed.op_type != "Transpose":
            continue
        weight = transposed.inputs[0]
        source = (weight.name or "").removesuffix(".weight")
        if not source.endswith((".attn1.to_q", ".attn1.to_v", ".attn2.to_q", ".attn2.to_v")):
            continue
        width, output = weight.shape[1], weight.shape[0]
        a = ir.val(source + ".lora.A", dtype=ir.DataType.FLOAT, shape=[width, rank])
        b = ir.val(source + ".lora.B", dtype=ir.DataType.FLOAT, shape=[rank, output])
        if {a.name, b.name} & existing:
            raise ValueError(f"Adapter name collision: {source}")
        existing.update((a.name, b.name))
        result.graph.inputs.extend((a, b))
        rng = np.random.default_rng(20261102 + len(params))
        params[a.name] = rng.normal(0, 0.02, tuple(a.shape)).astype(np.float32)
        params[b.name] = np.zeros(tuple(b.shape), np.float32)
        users = list(node.outputs[0].uses())
        tape = ir.tape.Tape()
        low_rank = tape.op("MatMul", [tape.op("MatMul", [node.inputs[0], a]), b])
        scale = tape.op("Constant", [], {"value": ir.tensor(np.float32(alpha / rank))})
        adapted = tape.op("Add", [node.outputs[0], tape.op("Mul", [low_rank, scale])])
        for index, operation in enumerate(tape.nodes):
            operation.name = f"{source}.adapter.{index}"
            for index2, value in enumerate(operation.outputs):
                value.name = f"{operation.name}.{index2}"
        result.graph.insert_after(node, tape.nodes)
        for use in users:
            use.node.replace_input_with(use.idx, adapted)
        mapping.append(
            {"module": source, "A": a.name, "B": b.name, "rank": rank, "alpha": alpha}
        )
    if len(mapping) != 8:
        raise ValueError("Expected q/v adapters in both attention blocks of both layers")
    ShapeInferencePass()(result)
    return result, params, mapping
