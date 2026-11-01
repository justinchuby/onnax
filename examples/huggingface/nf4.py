# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Store Qwen decoder Linear weights as packed NF4 with FP32 adapter training."""

import numpy as np
import torch

from examples.lora.nf4 import CODEBOOK, dequantize, lower, quantize


def lower_decoder(model, reference, *, block_size=64, precision="fp32"):
    names = sorted(
        {
            node.inputs[1].name
            for node in model.graph
            if not node.domain
            and node.op_type == "MatMul"
            and node.inputs[1].const_value is not None
            and (node.inputs[1].name or "").startswith("model.layers.")
            and node.inputs[1].name.endswith(".weight_t")
        }
    )
    expected = reference.config.num_hidden_layers * 7
    if len(names) != expected:
        raise ValueError(
            f"Expected {expected} Qwen decoder Linear weights, found {len(names)}"
        )
    quantized, manifest = lower(model, names=names, block_size=block_size, precision=precision)
    # Replay the exact stored quantization in the independent Transformers model.
    # ONNX stores Linear's transposed matrix in row-major NF4 blocks.
    with torch.no_grad():
        for name, row in manifest.items():
            values = quantized.graph.initializers
            decoded = dequantize(
                values[row["packed"]].const_value.numpy(),
                values[row["scales"]].const_value.numpy(),
                row["shape"],
                block_size=row["block_size"],
            )
            module = reference.get_submodule(name.removesuffix(".weight_t"))
            module.weight.copy_(torch.from_numpy(np.ascontiguousarray(decoded.T)))
    return quantized, {
        "format": "onnax-row-major-nf4-v1",
        "codebook": CODEBOOK.tolist(),
        "packing": "high nibble first; row-major transposed Linear weight",
        "block_size": block_size,
        "compute_dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision],
        "adapter_dtype": "float32",
        "embedding_dtype": "float32",
        "lm_head_dtype": "float32",
        "weights": manifest,
        "original_decoder_bytes": sum(row["original_bytes"] for row in manifest.values()),
        "packed_decoder_bytes": sum(
            row["packed_bytes"] + row["scale_bytes"] for row in manifest.values()
        ),
    }


def quantize_reference(reference, *, block_size=64):
    """Use identical weight values in isolated PyTorch measurements."""
    count = 0
    with torch.no_grad():
        for name, module in reference.named_modules():
            if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().numpy().T
                packed, scales = quantize(weight, block_size=block_size)
                decoded = dequantize(packed, scales, weight.shape, block_size=block_size)
                module.weight.copy_(torch.from_numpy(np.ascontiguousarray(decoded.T)))
                count += 1
    if count != reference.config.num_hidden_layers * 7:
        raise ValueError("Expected seven frozen Linear weights per Qwen decoder layer")
