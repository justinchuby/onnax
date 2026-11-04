# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure a Transformers reference using an exported ONNX task's adapter snapshot."""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path  # ruff: ignore[banned-api] - Standalone benchmark paths.

import numpy as np
import torch

from examples.huggingface.benchmark import hardware
from examples.huggingface.decoder_families import reference_model
from examples.huggingface.mixed_precision import reference_projections
from examples.huggingface.nf4 import quantize_reference
from examples.huggingface.torch_context import (
    checkpoint_decoder,
    chunked_token_loss,
    synchronize,
)
from examples.huggingface.validation import add_reference_parameters
from examples.lora.nf4 import CODEBOOK


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_arrays(arrays):
    """Hash names, dtypes, shapes and values independently of archive encoding."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def reference_from_export(directory, *, cache_dir=None):
    """Read the named adapter snapshot without constructing ONNX runtime sessions."""
    directory = Path(directory)
    task = json.loads((directory / "task.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if task.get("version") not in (1, 2):
        raise ValueError("Expected an exported decoder task")
    quantization = task.get("quantization")
    if quantization != config.get("onnax_quantization"):
        raise ValueError("Decoder quantization differs from the saved configuration")
    dtype = config.get("onnax_precision", {}).get("projection_compute", "float32")
    if dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("Unknown projection compute dtype")
    if quantization is not None:
        if (
            quantization.get("format") != "onnax-row-major-nf4-v1"
            or quantization.get("codebook") != CODEBOOK.tolist()
            or quantization.get("compute_dtype") != dtype
            or isinstance(quantization.get("block_size"), bool)
            or not isinstance(quantization.get("block_size"), int)
            or quantization["block_size"] < 1
        ):
            raise ValueError("Unsupported NF4 reference policy")
    execution = directory / "execution"
    interface = json.loads((execution / "interface.json").read_text())
    if interface.get("version") != 1 or interface.get("kind") != "checkpointed_vjp":
        raise ValueError("Expected a checkpointed decoder bundle")
    if _digest(execution / "parameters.npz") != interface["files"]["parameters.npz"]:
        raise ValueError("Adapter snapshot integrity mismatch")
    names = interface["specification"]["parameters"]
    with np.load(execution / "parameters.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"arr_{i}" for i in range(len(names))}:
            raise ValueError("Adapter archive differs from the parameter interface")
        arrays = {name: archive[f"arr_{i}"].copy() for i, name in enumerate(names)}
    mapped = [row[key] for row in task["mapping"] for key in ("A", "B")]
    if len(set(mapped)) != len(mapped) or set(mapped) != set(arrays):
        raise ValueError("Adapter mapping differs from the parameter interface")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Expected finite FP32 adapters")
    reference = reference_model(
        task["family"], real=bool(config.get("onnax_pretrained")), cache_dir=cache_dir
    )
    for field in (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "layer_types",
    ):
        if config.get(field) != getattr(reference.config, field, None):
            raise ValueError(f"Source configuration differs: {field}")
    if config.get("onnax_pretrained") != getattr(reference.config, "onnax_pretrained", None):
        raise ValueError("Pinned pretrained source differs")
    for row in task["mapping"]:
        layer = reference.get_submodule(row["module"])
        rank = row["rank"]
        if arrays[row["A"]].shape != (layer.in_features, rank) or arrays[row["B"]].shape != (
            rank,
            layer.out_features,
        ):
            raise ValueError("Adapter shape differs from its source projection")
    if quantization is not None:
        quantize_reference(
            reference, mapping=task["mapping"], block_size=quantization["block_size"]
        )
    add_reference_parameters(reference, arrays, task["mapping"])
    if dtype != "float32":
        reference_projections(
            reference, precision={"bfloat16": "bf16", "float16": "fp16"}[dtype]
        )
    reference.config.onnax_precision = config.get("onnax_precision", {})
    return reference, task, arrays


def measure(
    directory,
    batch,
    *,
    device="cpu",
    attention="sdpa",
    checkpoint_layers=1,
    head_chunk_size=128,
    warmup=10,
    samples=100,
    cache_dir=None,
):
    """Time fixed-parameter forward/loss/backward with explicit device boundaries."""
    if device not in ("cpu", "mps") or attention not in ("eager", "sdpa"):
        raise ValueError("Expected CPU/MPS and eager/SDPA attention")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable MPS CPU fallback for device measurements")
    if (
        isinstance(head_chunk_size, bool)
        or not isinstance(head_chunk_size, int)
        or head_chunk_size < 1
    ):
        raise ValueError("Expected a positive head chunk size")
    if warmup < 0 or samples < 1:
        raise ValueError("Expected nonnegative warmup and positive samples")
    if set(batch) != {"input_ids", "attention_mask", "position_ids", "labels"}:
        raise ValueError("Expected token IDs, attention mask, position IDs and labels")
    ids = batch["input_ids"]
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 2:
        raise ValueError("Expected nonempty batched sequences with at least two tokens")
    if any(v.dtype != np.int64 or v.shape != ids.shape for v in batch.values()):
        raise ValueError("Expected same-shaped INT64 inputs")
    if not np.isin(batch["attention_mask"], [0, 1]).all() or (batch["position_ids"] < 0).any():
        raise ValueError("Expected a binary attention mask and nonnegative position IDs")
    source_files = [
        Path(__file__),
        Path(__file__).with_name("torch_context.py"),
        Path(__file__).with_name("decoder_families.py"),
        Path(__file__).with_name("validation.py"),
        Path(__file__).with_name("mixed_precision.py"),
        Path(__file__).with_name("nf4.py"),
        Path(__file__).parents[1] / "lora" / "nf4.py",
        Path(__file__).parents[1] / "torch_precision.py",
        Path(__file__).with_name("pretrained.py"),
        Path(__file__).with_name("qwen35.py"),
        Path(directory) / "task.json",
        Path(directory) / "config.json",
        Path(directory) / "execution" / "interface.json",
        Path(directory) / "execution" / "parameters.npz",
    ]
    source_hashes = {str(path): _digest(path) for path in source_files}
    torch.set_num_threads(1)
    start = time.perf_counter()
    model, task, initial = reference_from_export(directory, cache_dir=cache_dir)
    if ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Sequence exceeds the source context limit")
    if (ids < 0).any() or (ids >= model.config.vocab_size).any():
        raise ValueError("Token IDs exceed the vocabulary")
    labels = batch["labels"]
    if ((labels != -100) & ((labels < 0) | (labels >= model.config.vocab_size))).any():
        raise ValueError("Labels exceed the vocabulary")
    model.set_attn_implementation(attention)
    # Keep references to the original parameters: wrapping changes module paths.
    parameters = {
        row[key]: getattr(model.get_submodule(row["module"]), key)
        for row in task["mapping"]
        for key in ("A", "B")
    }
    model = checkpoint_decoder(model, checkpoint_layers).to(device)
    values = {name: torch.from_numpy(array.copy()).to(device) for name, array in batch.items()}
    labels = values.pop("labels")
    synchronize(device)
    construction = time.perf_counter() - start
    records = []
    losses = []
    for index in range(warmup + samples):
        model.zero_grad(set_to_none=True)
        synchronize(device)
        before = time.perf_counter()
        loss = chunked_token_loss(model, values, labels, head_chunk_size)
        loss.backward()
        synchronize(device)
        elapsed = time.perf_counter() - before
        # Device exchange is measured separately from device-resident execution.
        before = time.perf_counter()
        gradients = {
            name: parameter.grad.detach().cpu().numpy().copy()
            for name, parameter in parameters.items()
        }
        loss_value = loss.detach().cpu().numpy().copy()
        synchronize(device)
        exchange = time.perf_counter() - before
        if not np.isfinite(loss_value).all() or any(
            not np.isfinite(g).all() for g in gradients.values()
        ):
            raise FloatingPointError("Nonfinite loss or adapter gradients")
        if index >= warmup:
            losses.append(float(loss_value))
            records.append(
                {
                    "execution_seconds": elapsed,
                    "output_exchange_seconds": exchange,
                    "total_seconds": elapsed + exchange,
                }
            )
    final = {name: p.detach().cpu().numpy() for name, p in parameters.items()}
    if digest_arrays(final) != digest_arrays(initial):
        raise AssertionError("Fixed adapter snapshot changed")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "version": 1,
        "family": task["family"],
        "device": device,
        "attention": attention,
        "checkpoint_layers": checkpoint_layers,
        "head_chunk_size": head_chunk_size,
        "warmup": warmup,
        "losses": losses,
        "input_shape": list(ids.shape),
        "gradient_count": len(gradients),
        "samples": records,
        "dtype": "float32",
        "projection_compute": getattr(model.config, "onnax_precision", {}).get(
            "projection_compute", "float32"
        ),
        "frozen_weight_storage": (
            "NF4 values dequantized into FP32 reference parameters"
            if task.get("quantization") is not None
            else "float32"
        ),
        "autocast": False,
        "threads": 1,
        "effective_tokens": int((batch["labels"][:, 1:] != -100).sum()),
        "input_sha256": digest_arrays(batch),
        "adapter_sha256": digest_arrays(initial),
        "construction_seconds": construction,
        "timings": {
            key: {
                "median": float(np.median([r[key] for r in records])),
                "p95": float(np.percentile([r[key] for r in records], 95)),
            }
            for key in records[0]
        },
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "timing_scope": "Fixed-parameter forward, mean shifted-token loss and all adapter gradients; excludes optimizer",
        "exchange_scope": "Inputs staged once before warmup; gradients/loss copied to CPU each iteration",
        "memory_scope": "Independent worker, reference construction, adapters and execution",
        "head_schedule": "Non-reentrant checkpointed frozen projection and native cross entropy",
        "source": getattr(model.config, "onnax_pretrained", None),
        "hardware": hardware(),
        "versions": {name: version(name) for name in ("torch", "transformers", "numpy")},
        "platform": sys.platform,
        "mps_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            )
        },
    }
    if any(_digest(path) != source_hashes[str(path)] for path in source_files):
        raise RuntimeError("Measurement source or adapter snapshot changed")
    report["source_sha256"] = source_hashes
    report["mps_cpu_fallback"] = False if device == "mps" else None
    return report, {**gradients, "loss": loss_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument(
        "--inputs", required=True, help="NPZ with IDs, mask, positions and labels"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--checkpoint-layers", type=int, default=1)
    parser.add_argument("--head-chunk-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as archive:
        batch = {name: archive[name].copy() for name in archive.files}
    report, gradients = measure(
        args.export,
        batch,
        device=args.device,
        attention=args.attention,
        checkpoint_layers=args.checkpoint_layers,
        head_chunk_size=args.head_chunk_size,
        warmup=args.warmup,
        samples=args.samples,
        cache_dir=args.cache_dir,
    )
    np.savez(args.output + ".npz", **gradients)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timings"]))


if __name__ == "__main__":
    main()
