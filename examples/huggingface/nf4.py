# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Store decoder Linear weights as packed NF4 with FP32 adapter training."""

import numpy as np
import torch

from examples.lora.nf4 import CODEBOOK, dequantize, lower, quantize


def _linears(reference):
    modules = {
        name: module
        for name, module in reference.named_modules()
        if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)
    }
    if not modules:
        raise ValueError("Expected decoder Linear modules")
    return modules


def _pieces(reference, mapping):
    modules = _linears(reference)
    pieces = {}
    seen = set()
    for row in mapping:
        source = row["module"]
        if source not in modules or source in seen:
            raise ValueError(f"Missing or duplicate source Linear: {source}")
        seen.add(source)
        end = 0
        for piece in row["pieces"]:
            name = piece["module"] + ".weight_t"
            start, stop = piece["start"], piece["stop"]
            if name in pieces or start != end or stop <= start:
                raise ValueError("Source projection slices must be unique and contiguous")
            pieces[name] = (source, start, stop)
            end = stop
        if end != modules[source].weight.shape[0]:
            raise ValueError(f"Incomplete source projection slices: {source}")
    return modules, pieces


def lower_decoder(model, reference, *, mapping=(), block_size=64, precision="fp32"):
    """Pack frozen graph projections and reconstruct their native source slices."""
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
    modules, pieces = _pieces(reference, mapping)
    sources, coverage = {}, {}
    for name in names:
        direct = name.removesuffix(".weight_t")
        source, start, stop = pieces.get(
            name, (direct, 0, modules[direct].weight.shape[0] if direct in modules else 0)
        )
        if source not in modules:
            raise ValueError(f"Missing source Linear mapping: {name}")
        weight = modules[source].weight.detach().cpu().numpy()[start:stop].T
        stored = model.graph.initializers[name].const_value.numpy()
        if not np.array_equal(weight, stored):
            raise ValueError(f"Source Linear values differ from the ONNX projection: {name}")
        sources[name] = (source, start, stop)
        coverage.setdefault(source, []).append((start, stop))
    if set(coverage) != set(modules):
        raise ValueError("Frozen ONNX projections must cover every source decoder Linear")
    for source, intervals in coverage.items():
        end = 0
        for start, stop in sorted(intervals):
            if start != end or stop <= start:
                raise ValueError(f"Overlapping or incomplete source Linear: {source}")
            end = stop
        if end != modules[source].weight.shape[0]:
            raise ValueError(f"Incomplete source Linear: {source}")
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
            source, start, stop = sources[name]
            modules[source].weight[start:stop].copy_(
                torch.from_numpy(np.ascontiguousarray(decoded.T))
            )
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


def quantize_reference(reference, *, mapping=(), block_size=64):
    """Reconstruct source weights with the graph's per-projection NF4 blocks."""
    modules, pieces = _pieces(reference, mapping)
    split = {}
    for source, start, stop in pieces.values():
        split.setdefault(source, []).append((start, stop))
    with torch.no_grad():
        for name, module in modules.items():
            intervals = sorted(split.get(name, [(0, module.weight.shape[0])]))
            for start, stop in intervals:
                weight = module.weight[start:stop].detach().cpu().numpy().T
                packed, scales = quantize(weight, block_size=block_size)
                decoded = dequantize(packed, scales, weight.shape, block_size=block_size)
                module.weight[start:stop].copy_(
                    torch.from_numpy(np.ascontiguousarray(decoded.T))
                )
