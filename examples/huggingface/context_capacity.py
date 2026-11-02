# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Record one real Qwen capacity step and optional construction/session memory phases."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time

import torch

from examples.huggingface.benchmark import measure
from examples.process_memory import physical_footprint


class MemoryJournal:
    """Write process and allocator snapshots at explicit execution boundaries."""

    def __init__(self, path, *, provider, library=None):
        self.path, self.provider = path, provider
        self.start = time.perf_counter()
        self.api = ctypes.CDLL(library) if library else None
        if self.api is not None:
            for name in ("active", "cache", "peak"):
                function = getattr(self.api, f"mlx_get_{name}_memory")
                function.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
                function.restype = ctypes.c_int

    def __call__(self, phase, **details):
        row = {
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - self.start,
            "physical_footprint": physical_footprint(),
            **details,
        }
        if self.api is not None:
            for name in ("active", "cache", "peak"):
                value = ctypes.c_size_t()
                if getattr(self.api, f"mlx_get_{name}_memory")(ctypes.byref(value)) != 0:
                    raise RuntimeError("Could not read MLX allocator memory")
                row[f"mlx_{name}_bytes"] = value.value
        if self.provider == "mps":
            row["mps_active_bytes"] = torch.mps.current_allocated_memory()
            row["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def run(
    directory,
    *,
    provider,
    train_file,
    library=None,
    projection_precision="bf16",
    sequence_length=8192,
    checkpoint_layers=4,
    chunk_size=128,
    trace_phases=False,
):
    """Validate a cold update; steady-state timing uses the separate benchmark CLI."""
    if provider not in ("cpu", "mlx", "mps"):
        raise ValueError("Expected cpu, mlx or mps provider")
    if provider == "mlx" and not library:
        raise ValueError("MLX capacity qualification requires an explicit library")
    if provider != "mlx" and library:
        raise ValueError("A native library applies only to the MLX provider")
    if provider == "cpu" and projection_precision != "fp32":
        raise ValueError("CPU capacity qualification uses FP32 projections")
    os.makedirs(directory, exist_ok=False)
    journal = MemoryJournal(
        os.path.join(directory, "phases.jsonl"), provider=provider, library=library
    )
    journal("qualification_begin")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    report = measure(
        "pytorch" if provider == "mps" else "chunked",
        real=True,
        provider="mlx" if provider == "mlx" else "cpu",
        library=library,
        torch_device="mps" if provider == "mps" else "cpu",
        torch_chunked=provider == "mps",
        torch_attention="sdpa" if provider == "mps" else "eager",
        nf4=True,
        fused_ops=provider == "mlx",
        projection_precision=projection_precision,
        batch_size=1,
        sequence_length=sequence_length,
        checkpoint_layers=checkpoint_layers,
        chunk_size=chunk_size,
        train_file=train_file,
        warmup=0,
        samples=1,
        mlx_memory_limit_bytes=20 * 1024**3 if provider == "mlx" else None,
        phase_observer=journal if trace_phases else None,
        validate_updates=True,
        torch_mps_memory_limit_bytes=20 * 1024**3 if provider == "mps" else None,
    )
    if report["parameter_tensors"] != 112:
        raise AssertionError("Expected all 112 Qwen adapter tensors")
    report.update(
        source_commit=source,
        working_tree_dirty=dirty,
        qualification_scope="Cold one-update capacity, with FP32 parameter/gradient/AdamW validation",
        phase_journal="phases.jsonl",
    )
    with open(train_file, "rb") as stream:
        report["input_file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    journal("qualification_complete")
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=("cpu", "mlx", "mps"), required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--library")
    parser.add_argument(
        "--projection-precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--checkpoint-layers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--trace-phases", action="store_true")
    args = parser.parse_args()
    report = run(
        args.output_dir,
        provider=args.provider,
        library=args.library,
        train_file=args.train_file,
        projection_precision=args.projection_precision,
        sequence_length=args.sequence_length,
        checkpoint_layers=args.checkpoint_layers,
        chunk_size=args.chunk_size,
        trace_phases=args.trace_phases,
    )
    print(
        json.dumps(
            {
                "loss": report["losses"],
                "physical_footprint": report["physical_footprint"],
                "parameter_tensors": report["parameter_tensors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
