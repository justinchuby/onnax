# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent PyTorch equations for the offline conditional DiT workload."""

import math

import torch
from torch.nn import functional


def forward(config, weights, adapters, mapping, sample, timestep, encoder_hidden_states):
    """Evaluate denoising directly, without executing or translating an ONNX graph."""
    by_module = {row["module"]: row for row in mapping}

    def linear(x, name):
        result = functional.linear(x, weights[name + ".weight"], weights[name + ".bias"])
        if name in by_module:
            row = by_module[name]
            result = result + (row["alpha"] / row["rank"]) * (
                x @ adapters[row["A"]] @ adapters[row["B"]]
            )
        return result

    def norm(x, name):
        return functional.layer_norm(
            x,
            (config.hidden_size,),
            weights[name + ".weight"],
            weights[name + ".bias"],
            config.norm_eps,
        )

    def attention(x, context, name):
        q, k, v = (
            linear(x, name + ".to_q"),
            linear(context, name + ".to_k"),
            linear(context, name + ".to_v"),
        )
        b, length, _ = q.shape
        heads, dim = (
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )
        q = q.reshape(b, length, heads, dim).transpose(1, 2)
        k = k.reshape(b, -1, heads, dim).transpose(1, 2)
        v = v.reshape(b, -1, heads, dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * (dim**-0.5)
        output = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, length, -1)
        return linear(output, name + ".to_out.0")

    p = config.patch_size
    batch, _, height, width = sample.shape
    x = functional.conv2d(
        sample, weights["pos_embed.proj.weight"], weights["pos_embed.proj.bias"], stride=p
    )
    x = x.flatten(2).transpose(1, 2)
    half = config.hidden_size // 2
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float64) * (-math.log(10000) / half)
    ).float()
    angles = timestep.float()[:, None] * frequencies[None]
    temb = linear(
        functional.silu(
            linear(torch.cat((angles.cos(), angles.sin()), dim=-1), "adaln_single.linear_1")
        ),
        "adaln_single.linear_2",
    )
    context = linear(encoder_hidden_states, "caption_projection")
    for index in range(config.num_layers):
        name = f"transformer_blocks.{index}"
        shift1, scale1, gate1, shift2, scale2, gate2 = linear(
            functional.silu(temb), name + ".norm1.linear"
        ).chunk(6, dim=-1)
        attn_input = norm(x, name + ".norm1.norm") * (1 + scale1[:, None]) + shift1[:, None]
        x = x + attention(attn_input, attn_input, name + ".attn1") * gate1[:, None]
        x = x + attention(norm(x, name + ".attn2.norm"), context, name + ".attn2")
        ff_input = norm(x, name + ".norm1.norm") * (1 + scale2[:, None]) + shift2[:, None]
        ff_output = linear(
            functional.gelu(linear(ff_input, name + ".ff.linear_1")), name + ".ff.linear_2"
        )
        x = x + ff_output * gate2[:, None]
    shift, scale = linear(functional.silu(temb), "norm_out.linear").chunk(2, dim=-1)
    x = norm(x, "norm_out.norm") * (1 + scale[:, None]) + shift[:, None]
    x = linear(x, "proj_out")
    return (
        x.reshape(batch, height // p, width // p, p, p, config.out_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, config.out_channels, height, width)
    )
