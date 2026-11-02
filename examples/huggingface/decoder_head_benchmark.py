# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure one decoder loss/VJP route in an independent worker."""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version

import numpy as np
import torch

from examples.huggingface.benchmark import hardware
from examples.huggingface.decoder_sft import prepare


def source_digests():
    result = {}
    for name in (
        "decoder_head_benchmark.py",
        "decoder_sft.py",
        "chunked_loss.py",
        "decoder_families.py",
        "qwen35.py",
        "decoder_head.py",
    ):
        with open(os.path.join(os.path.dirname(__file__), name), "rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["phi3", "qwen3", "qwen35"], default="qwen35")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--head-chunk-size", type=int)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--valid-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.sequence < 2
        or not 1 <= args.valid_tokens < args.sequence
        or args.warmup < 0
        or args.samples < 1
    ):
        parser.error(
            "Require sequence >= 2, 1 <= valid tokens < sequence and positive samples"
        )
    if args.library and args.provider != "mlx":
        parser.error("--library requires MLX")
    factory = None
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if args.library:
            register_library(args.library)
        factory = session
    sources = source_digests()
    torch.set_num_threads(1)
    start = time.perf_counter()
    model, reference = prepare(
        args.family,
        session_factory=factory,
        real=args.real,
        cache_dir=args.cache_dir,
        fused_ops=True,
        head_chunk_size=args.head_chunk_size,
    )
    if args.sequence > model.config.max_position_embeddings:
        model.close()
        parser.error("Sequence exceeds the source model context limit")
    del reference
    gc.collect()
    construction = time.perf_counter() - start
    rng = np.random.default_rng(20261102)
    ids = torch.from_numpy(rng.integers(3, model.config.vocab_size, (1, args.sequence)))
    labels = torch.full_like(ids, -100)
    selected = rng.choice(args.sequence - 1, args.valid_tokens, replace=False) + 1
    labels[0, selected] = ids[0, selected]
    records = []
    try:
        # Nonzero B exercises both adapter derivative branches at fixed parameters.
        with torch.no_grad():
            for name, parameter in model.onnx.parameters_by_name().items():
                if name.endswith(".B"):
                    parameter.copy_(
                        torch.from_numpy(
                            rng.normal(0, 0.01, tuple(parameter.shape)).astype(np.float32)
                        )
                    )
        for index in range(args.warmup + args.samples):
            model.zero_grad(set_to_none=True)
            before = time.perf_counter()
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            elapsed = time.perf_counter() - before
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Nonfinite loss at iteration {index}")
            if (index + 1) % 10 == 0:
                print(
                    json.dumps({"iteration": index + 1, "loss": float(loss.detach())}),
                    flush=True,
                )
            if index >= args.warmup:
                records.append(
                    {
                        "step_seconds": elapsed,
                        **(model.head.last_profile if model.head is not None else {}),
                    }
                )
        arrays = {
            name: parameter.grad.numpy().copy()
            for name, parameter in model.onnx.parameters_by_name().items()
        }
        arrays["loss"] = loss.detach().numpy()
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite decoder loss or adapter gradients")
        np.savez(args.output + ".npz", **arrays)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            "version": 1,
            "options": vars(args),
            "samples": records,
            "timings": {
                key: {
                    "median": float(np.median([r[key] for r in records])),
                    "p95": float(np.percentile([r[key] for r in records], 95)),
                }
                for key in records[0]
                if key.endswith("seconds")
            },
            "construction_seconds": construction,
            "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "memory_scope": "Independent worker, imports, source construction, sessions and execution",
            "timing_scope": "Fixed-parameter forward, loss, backward and runtime exchange; excludes optimizer",
            "synchronization": "Synchronous runtime outputs and CPU gradients before timer stops",
            "input_sha256": hashlib.sha256(
                ids.numpy().tobytes() + labels.numpy().tobytes()
            ).hexdigest(),
            "source_identity": model.onnx._identity,
            "ort_graph_optimization": "disabled",
            "threads": 1,
            "cpu_fallback": "disabled" if args.provider == "mlx" else None,
            "hardware": hardware(),
            "versions": {
                name: version(name)
                for name in (
                    "onnax",
                    "onnx",
                    "onnx-ir",
                    "onnxscript",
                    "onnxruntime",
                    "torch",
                    "mobius-onnx",
                )
            },
            "source_sha256": sources,
            "dtype": "float32",
            "platform": platform.platform(),
            "runtime_environment": {
                key: os.environ.get(key)
                for key in (
                    "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                    "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                )
            },
        }
        if source_digests() != sources:
            raise RuntimeError("Benchmark source files changed during measurement")
        if args.library:
            with open(args.library, "rb") as stream:
                report["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report["timings"]))
    finally:
        model.close()


if __name__ == "__main__":
    main()
