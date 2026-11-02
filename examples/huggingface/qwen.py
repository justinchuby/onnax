# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Construct cache-free Qwen3 ONNX graphs with Mobius and inject explicit LoRA."""

import numpy as np
import onnx_ir as ir
import torch
from mobius.models import Qwen3CausalLMModel
from onnx_ir.passes.common import ShapeInferencePass
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.huggingface.decoder_graph import (
    DECODER_BOUNDARIES as DECODER_BOUNDARIES,
)
from examples.huggingface.decoder_graph import (
    TrainingTask as TrainingTask,
)
from examples.huggingface.decoder_graph import (
    build_base as build_decoder_base,
)
from examples.huggingface.decoder_graph import (
    decompose as decompose,
)
from examples.huggingface.decoder_graph import (
    protect_decoder_boundaries as protect_decoder_boundaries,
)
from examples.huggingface.decoder_graph import (
    restore_decoder_outputs as restore_decoder_outputs,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def tiny_config():
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        attention_dropout=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )


def build_base(reference):
    return build_decoder_base(reference, Qwen3CausalLMModel)


def reference_model(real=False):
    torch.manual_seed(20261029)
    if real:
        result = Qwen3ForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        result.config._commit_hash = REVISION
        return result
    config = tiny_config()
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def inject(
    model,
    *,
    rank=4,
    alpha=1.0,
    target_suffixes=(".self_attn.q_proj", ".self_attn.v_proj"),
):
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
        or not np.isfinite(alpha)
    ):
        raise ValueError("Expected a positive integer rank and finite alpha")
    result = model.clone()
    existing = {
        v.name
        for v in (
            *result.graph.inputs,
            *result.graph.initializers.values(),
            *(v for n in result.graph for v in n.outputs),
        )
    }
    params, mapping = {}, []
    for node in list(result.graph):
        if node.domain or node.op_type != "MatMul":
            continue
        rhs = node.inputs[1]
        # Mobius folds Linear's transpose into a source-named initializer.
        producer = rhs.producer()
        if rhs.const_value is not None and (rhs.name or "").endswith(".weight_t"):
            module = rhs.name.removesuffix(".weight_t")
            in_width, out_width = rhs.shape
        elif producer is not None and producer.op_type == "Transpose":
            weight = producer.inputs[0]
            module = (weight.name or "").removesuffix(".weight")
            out_width, in_width = weight.shape
        else:
            continue
        if not module.endswith(target_suffixes):
            continue
        if {module + ".lora.A", module + ".lora.B"}.intersection(existing):
            raise ValueError(f"Adapter input name collision: {module}")
        a = ir.val(module + ".lora.A", dtype=ir.DataType.FLOAT, shape=[in_width, rank])
        b = ir.val(module + ".lora.B", dtype=ir.DataType.FLOAT, shape=[rank, out_width])
        result.graph.inputs.extend([a, b])
        rng = np.random.default_rng(20261029 + len(params))
        params[a.name] = rng.normal(0, 0.02, a.shape).astype(np.float32)
        params[b.name] = np.zeros(b.shape, np.float32)
        users = list(node.outputs[0].uses())
        tape = ir.tape.Tape()
        t = tape.op("MatMul", [node.inputs[0], a])
        d = tape.op("MatMul", [t, b])
        scale = tape.op("Constant", [], {"value": ir.tensor(np.float32(alpha / rank))})
        delta = tape.op("Mul", [d, scale])
        output = tape.op("Add", [node.outputs[0], delta])
        for i, new_node in enumerate(tape.nodes):
            new_node.name = f"{module}.adapter.{i}"
            for j, value in enumerate(new_node.outputs):
                value.name = f"{module}.adapter.{i}.{j}"
        result.graph.insert_after(node, tape.nodes)
        for use in users:
            use.node.replace_input_with(use.idx, output)
        mapping.append(
            {"module": module, "A": a.name, "B": b.name, "rank": rank, "alpha": alpha}
        )
    if not mapping:
        raise ValueError("No source projection weights match the requested adapter targets")
    ShapeInferencePass()(result)
    return result, params, mapping
