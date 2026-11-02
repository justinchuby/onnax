# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure a frozen output head at a fixed causal supervision density."""

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path  # ruff: ignore[banned-api] - Measurement CLI output.

import numpy as np
import torch

from examples.huggingface.benchmark import hardware
from examples.huggingface.chunked_loss import ChunkedHead
from examples.lora.train import session


def dense(head, hidden, labels):
    """Execute the preceding scheduler, including ignored rows in projection."""
    array = hidden.numpy()
    rows = array[:, :-1].reshape(-1, array.shape[-1])
    targets = labels[:, 1:].contiguous().numpy().reshape(-1)
    gradient = np.zeros_like(rows)
    total = np.float32(0)
    for start in range(0, len(rows), head.chunk_size):
        stop = min(start + head.chunk_size, len(rows))
        value, derivative = head.executor.run(
            None, {"hidden": rows[start:stop], "labels": targets[start:stop]}
        )
        total += np.float32(value)
        gradient[start:stop] = derivative
    full = np.zeros_like(array)
    full[:, :-1] = gradient.reshape(array.shape[0], array.shape[1] - 1, array.shape[2])
    count = max(int(labels[:, 1:].ne(-100).sum()), 1)
    return total / count, full / count


class _DenseLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head, hidden, labels):
        value, gradient = dense(head, hidden.detach(), labels)
        ctx.save_for_backward(torch.from_numpy(gradient))
        return torch.tensor(value, dtype=torch.float32, device=hidden.device)

    @staticmethod
    def backward(ctx, upstream):
        return None, ctx.saved_tensors[0] * upstream, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--policy", choices=["gather", "dense"], default="gather")
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--valid-rows", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if (
        args.sequence < 2
        or min(args.width, args.vocab, args.samples) < 1
        or args.warmup < 0
        or not 0 <= args.valid_rows < args.sequence
    ):
        parser.error("Expected positive dimensions/samples and 0 <= valid rows < sequence")
    if args.library and args.provider != "mlx":
        parser.error("--library requires --provider mlx")
    factory = session
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        if args.library:
            register_library(args.library)
        factory = mlx_session
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261102)
    weight = rng.normal(0, 0.02, (args.width, args.vocab)).astype(np.float32)
    hidden = torch.from_numpy(
        rng.normal(size=(1, args.sequence, args.width)).astype(np.float32)
    )
    labels = torch.full((1, args.sequence), -100, dtype=torch.int64)
    chosen = rng.choice(args.sequence - 1, args.valid_rows, replace=False) + 1
    labels[0, chosen] = torch.from_numpy(rng.integers(args.vocab, size=args.valid_rows))
    before = time.perf_counter()
    head = ChunkedHead(weight, chunk_size=args.chunk_size, session_factory=factory)
    construction = time.perf_counter() - before

    def gather():
        value = hidden.detach().requires_grad_()
        loss = head(value, labels)
        loss.backward()
        return float(loss.detach()), value.grad.numpy()

    # Compare both schedulers in each independent worker before timing.
    expected = dense(head, hidden, labels)
    actual = gather()
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-6)

    def dense_step():
        value = hidden.detach().requires_grad_()
        loss = _DenseLoss.apply(head, value, labels)
        loss.backward()

    run = gather if args.policy == "gather" else dense_step
    records = []
    for index in range(args.warmup + args.samples):
        before = time.perf_counter()
        run()
        elapsed = time.perf_counter() - before
        if index >= args.warmup:
            records.append(
                {
                    "step_seconds": elapsed,
                    **(head.last_profile if args.policy == "gather" else {}),
                }
            )
    timings = {
        key: {
            "median": float(np.median([row[key] for row in records])),
            "p95": float(np.percentile([row[key] for row in records], 95)),
        }
        for key in records[0]
        if key.endswith("seconds")
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "version": 1,
        "options": vars(args),
        "timings": timings,
        "samples": records,
        "construction_seconds": construction,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "memory_scope": "Whole worker including graph construction and both scheduler checks",
        "timing_scope": "Frozen head loss/hidden gradient, host scheduling and boundary exchange",
        "correctness": "Loss and full hidden gradient match dense scheduler at rtol=1e-4, atol=1e-6",
        "weight_bytes": weight.nbytes,
        "causal_rows": args.sequence - 1,
        "valid_rows": args.valid_rows,
        "executed_rows": args.valid_rows if args.policy == "gather" else args.sequence - 1,
        "input_sha256": hashlib.sha256(
            hidden.numpy().tobytes() + labels.numpy().tobytes()
        ).hexdigest(),
        "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "ort_graph_optimization": "disabled",
        "cpu_threads": 1,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
            )
        },
        "platform": platform.platform(),
        "hardware": hardware(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("chunked_loss.py", "supervision_benchmark.py")
        },
        "versions": {
            name: version(name) for name in ("numpy", "torch", "onnxruntime", "onnxscript")
        },
    }
    if args.library:
        report["library_sha256"] = hashlib.sha256(Path(args.library).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
