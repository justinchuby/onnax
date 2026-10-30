# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Validate tiny NF4 Qwen forward/VJP on the native MLX execution provider."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform

import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.residuals import split_vjp
from examples.huggingface.sft import prepare
from examples.huggingface.validation import add_reference_adapters, audit
from examples.lora.mlx_runtime import session


def run(*, library=None):
    """Compare four independent AdamW updates with CPU fallback disabled."""
    if library is not None:
        from examples.lora.mlx_runtime import register_library

        register_library(library)
    torch.set_num_threads(1)
    model, reference = prepare(nf4=True, save_activations=True)
    forward, _, _ = split_vjp(model.onnx.model, model.onnx.vjp_model)
    model.onnx.forward_session = session(forward)
    model.onnx.backward_session = session(model.onnx.optimized_vjp)
    reference = add_reference_adapters(reference, model)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    independent = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    losses = []
    checks = []
    for _ in range(4):
        result = audit(model, reference, batch, sync=False)
        checks.append(result)
        losses.append(result["loss"])
        optimizer.step()
        independent.step()
        for row in model.mapping:
            for key in ("A", "B"):
                actual = model.onnx.adapters[model.onnx.names.index(row[key])]
                expected = getattr(reference.get_submodule(row["module"]), key)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    if losses[-1] >= losses[0]:
        raise AssertionError(f"Training loss did not decrease: {losses}")
    if frozen_hashes(model.onnx.model) != model.onnx.frozen:
        raise AssertionError("Frozen model buffers changed")
    library_hash = None
    if library is not None:
        with open(library, "rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "model": "tiny-qwen3-nf4",
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnxruntime-ep-mlx", "torch", "mlx")
        },
        "library": os.path.abspath(library) if library else "installed plugin",
        "library_sha256": library_hash,
        "checks": checks,
        "provider": "MLXExecutionProvider",
        "cpu_fallback_disabled": True,
        "ort_graph_optimization": "disabled",
        "runtime_dtype": "float32",
        "boundary": "CPU PyTorch loss, parameters and AdamW; NumPy session exchange",
        "adapter_tensors": len(model.onnx.adapters),
        "steps": len(losses),
        "losses": losses,
        "gradient_and_update_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "frozen_buffers_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Development MLX EP dylib")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(library=args.library)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
