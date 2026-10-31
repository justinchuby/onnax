# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare joint and loss-only evaluation at identical adapter states."""

import argparse
import gc
import hashlib
import json
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path  # ruff: ignore[banned-api] - Standalone measurement CLI.

import numpy as np
import torch

from examples.huggingface.benchmark import hardware
from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.qwen import REVISION
from examples.huggingface.sft import prepare


def measure(*, real=False, precision="fp32", samples=100, warmup=10, factory=None):
    if samples < 1 or warmup < 0:
        raise ValueError("Expected samples >= 1 and warmup >= 0")
    torch.set_num_threads(1)
    options = {} if factory is None else {"session_factory": factory}
    model, reference = prepare(
        real, mode="joint", nf4=True, projection_precision=precision, **options
    )
    del reference
    gc.collect()
    joint = model.onnx
    initial = joint.arrays()
    frozen = frozen_hashes(joint.gradient)
    rng = np.random.default_rng(20261031)
    ids = torch.from_numpy(rng.integers(3, model.config.vocab_size, (1, 64)))
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(64)[None],
        labels=ids.clone(),
    )
    padded = {name: value[:, :17].repeat(2, 1) for name, value in batch.items()}
    padded["attention_mask"][1, 9:] = 0
    padded["position_ids"][1, 9:] = 8
    padded["labels"][1, 9:] = -100
    ignored = {name: value[:, :1] for name, value in batch.items()}
    ignored["labels"] = torch.full_like(ignored["labels"], -100)
    checks = []
    try:
        for state in range(3):
            with torch.no_grad():
                for name, parameter in zip(joint.names, joint.adapters, strict=True):
                    value = initial[name].copy()
                    if state:
                        value += rng.normal(0, state * 0.002, value.shape).astype(np.float32)
                    parameter.copy_(torch.from_numpy(value))
            for shape, current in (("full", batch), ("padded", padded), ("ignored", ignored)):
                baseline = joint(**current).detach().numpy().copy()
                with torch.no_grad():
                    actual = joint(**current).numpy().copy()
                if baseline.tobytes() != actual.tobytes() or not np.isfinite(actual).all():
                    raise AssertionError(f"Loss-only mismatch at state {state}, {shape}")
                checks.append({"state": state, "batch": shape, "loss_sum": float(actual)})
        timings, components = {}, {}
        output_bytes = {}
        for path in ("joint", "loss_only"):
            records = []
            counters = joint.timings if path == "joint" else joint.evaluation_timings
            execution_key = "loss_gradient" if path == "joint" else "loss"
            offset = len(counters[execution_key]) + warmup
            for index in range(warmup + samples):
                before = time.perf_counter()
                with torch.set_grad_enabled(path == "joint"):
                    value = joint(**batch).detach()
                # ORT run returns host arrays; both paths are synchronized here.
                elapsed = time.perf_counter() - before
                if not torch.isfinite(value):
                    raise AssertionError("Nonfinite measured loss")
                if index >= warmup:
                    records.append(elapsed * 1000)
            timings[path] = {
                "median": float(np.median(records)),
                "p95": float(np.percentile(records, 95)),
            }
            components[path] = {
                key: {
                    "median": float(np.median(counters[key][offset:]) * 1000),
                    "p95": float(np.percentile(counters[key][offset:], 95) * 1000),
                }
                for key in ("exchange", execution_key)
            }
            output_bytes[path] = joint.last_output_bytes
        if frozen_hashes(joint.gradient) != frozen:
            raise AssertionError("Frozen weights changed")
        return {
            "model": "Qwen3-0.6B" if real else "tiny Qwen3",
            "revision": REVISION,
            "nf4": True,
            "projection_precision": precision,
            "samples": samples,
            "warmup": warmup,
            "checks": checks,
            "timings_ms": timings,
            "component_timings_ms": components,
            "exchange_scope": "Host input/adapter validation and NumPy view preparation; runtime transfers are included in execution",
            "runtime_output_bytes": output_bytes,
            "evaluation_session_creation_seconds": joint.evaluation_timings[
                "session_creation"
            ],
            "nodes": {
                "joint": len(joint.gradient.graph),
                "loss_only": len(joint.loss_model.graph),
            },
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "memory_scope": "Whole process including construction, qualification and both sessions",
            "timing_scope": "Synchronous wrapper call, input preparation and runtime output exchange; fixed final adapter state",
            "hardware": hardware(),
            "platform": platform.platform(),
            "versions": {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "onnxruntime",
                    "onnxscript",
                    "mobius-onnx",
                )
            },
            "ort_optimizer": "disabled",
            "torch_threads": 1,
            "frozen_weights_unchanged": True,
        }
    finally:
        joint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--provider", choices=("cpu", "reference", "mlx"), default="cpu")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.precision == "bf16" and args.provider == "cpu":
        parser.error("BF16 projections require --provider reference or mlx")
    if args.library and args.provider != "mlx":
        parser.error("--library requires --provider mlx")
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if args.library:
            register_library(str(args.library))
    elif args.provider == "reference":
        from examples.huggingface.serialization import reference_session as session
    else:
        from examples.lora.train import session
    report = measure(
        real=args.real,
        precision=args.precision,
        samples=args.samples,
        warmup=args.warmup,
        factory=session,
    )
    report["provider"] = args.provider
    if args.library:
        report["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
