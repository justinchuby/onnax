# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline, cache-free Qwen3.5 text backbone built with Mobius."""

import numpy as np
import onnx_ir as ir
import torch
from mobius import CausalLMConfig, ModelPackage, ep_registry
from mobius._build_context import build_context
from mobius.models.qwen35 import Qwen35CausalLMModel
from mobius.tasks._base import _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _register_linear_attention_functions,
    linear_attention_dims,
)
from onnx_ir.passes.common import ShapeInferencePass
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from examples.huggingface.qwen import inject


def reference_model():
    """Build the four-layer 3:1 recurrent/full-attention qualification model."""
    torch.manual_seed(20261102)
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return Qwen3_5ForCausalLM(config).eval()


def build_base(reference):
    """Reset convolution/recurrent state for every full-sequence training call."""
    config = CausalLMConfig.from_transformers(reference.config)
    # The pinned generic CausalLMConfig lacks this architecture-specific field.
    # Transformers l2norm computes x / sqrt(sum(x*x) + 1e-6).
    config.linear_qk_l2norm_eps = 1e-6
    module = Qwen35CausalLMModel(config)
    with build_context(ep_registry.require("default"), ir.DataType.FLOAT):
        graph, builder = _make_graph()
        inputs = {
            name: builder.input(name, dtype=ir.DataType.INT64, shape=["batch", "sequence"])
            for name in ("input_ids", "attention_mask", "position_ids")
        }
        op = builder.op
        batch = op.Shape(inputs["input_ids"], start=0, end=1)
        dims = linear_attention_dims(config)

        def zeros(shape):
            return op.ConstantOfShape(
                op.Concat(batch, op.Constant(value_ints=shape), axis=0),
                value=ir.tensor(np.asarray([0], np.float32)),
            )

        states = [
            (
                zeros([dims.conv_dim, dims.conv_kernel - 1]),
                zeros([dims.num_v_heads, dims.head_k_dim, dims.head_v_dim]),
            )
            if kind == "linear_attention"
            else None
            for kind in config.layer_types
        ]
        hidden, _ = module.model(op, **inputs, past_key_values=states)
        logits = module.lm_head(op, hidden)
        builder.add_output(logits, name="logits")
        builder.add_output(hidden, name="hidden_states")
        model = _make_model(graph)
        _register_linear_attention_functions(model, config)
        package = ModelPackage({"model": model}, config)
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    model = package["model"]
    model.metadata_props["onnax.example.qwen35.qk_l2norm_eps"] = "1e-6"
    ShapeInferencePass()(model)
    return model


def prepare_graph():
    """Adapt fused linear-layer QKV and full-attention Q/gate and V projections."""
    reference = reference_model()
    base = build_base(reference)
    graph, params, mapping = inject(
        base,
        target_suffixes=(
            ".linear_attn.in_proj_qkv",
            ".self_attn.q_proj",
            ".self_attn.v_proj",
        ),
    )
    return graph, params, mapping, reference
