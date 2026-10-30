# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Observe allocator lifetimes and verify training resumes after idle cache release."""

import argparse
import gc
import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np

from examples.lora.mlx_training import host
from examples.mlp import NAMES
from examples.mlp_microbatch import Workload
from examples.mlx_binding import synchronize
from examples.mlx_provider_memory import ProviderMemory
from examples.process_memory import physical_footprint


def state_arrays(params, state):
    return {
        **{n: host(p).copy() for n, p in params.items()},
        **{f"adam.first.{n}": host(p).copy() for n, p in state.first.items()},
        **{f"adam.second.{n}": host(p).copy() for n, p in state.second.items()},
        "adam.step": np.array(state.step, dtype=np.int64),
    }


def step_arrays(result):
    loss, grads, params, state, _ = result
    return {
        "loss": host(loss).copy(),
        **{f"gradient.{n}": host(g).copy() for n, g in grads.items()},
        **state_arrays(params, state),
    }


def equal_arrays(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError("Checkpoint/output names changed")
    for name, value in actual.items():
        other = expected[name]
        if (
            value.dtype != other.dtype
            or value.shape != other.shape
            or value.tobytes() != other.tobytes()
        ):
            raise AssertionError(f"Cache release changed {name}")
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(other)):
            raise AssertionError(f"Nonfinite checkpoint/output {name}")


def run(directory, *, precision="fp32", microbatch=16384, settle_ms=250):
    if (
        isinstance(settle_ms, bool)
        or not isinstance(settle_ms, (float, int))
        or not np.isfinite(settle_ms)
        or not 0 <= settle_ms <= 10000
    ):
        raise ValueError("Expected a finite settle interval between 0 and 10000 ms")
    import mlx.core as mx

    workload = Workload(
        directory, profile="activation", precision=precision, microbatch=microbatch
    )
    params, state = workload.parameters(), None
    provider = ProviderMemory()
    phases = []

    def snapshot(label):
        mx.synchronize()
        gc.collect()
        phases.append(
            {
                "phase": label,
                "provider": provider.read(),
                "python_mlx": {
                    "active_bytes": mx.get_active_memory(),
                    "cache_bytes": mx.get_cache_memory(),
                    "peak_bytes": mx.get_peak_memory(),
                },
                "physical_footprint": physical_footprint(),
            }
        )

    snapshot("setup")
    times = []
    for index in range(110):
        loss, grads, params, state, timings = workload.step(params, state)
        del loss, grads
        if index >= 10:
            times.append(timings["step"])
    synchronize(params, state)
    if state.step != 110:
        raise AssertionError("Baseline update count changed")
    before = state_arrays(params, state)
    baseline = workload.step(params, state)
    if baseline[3].step != 111:
        raise AssertionError("Expected update 111")
    expected = step_arrays(baseline)
    del baseline
    snapshot("before_release")
    mx.clear_cache()
    snapshot("after_python_clear")
    time.sleep(settle_ms / 1000)
    snapshot("after_python_settle")
    provider.clear_cache()
    snapshot("after_provider_clear")
    time.sleep(settle_ms / 1000)
    snapshot("after_provider_settle")
    equal_arrays(state_arrays(params, state), before)
    resumed = workload.step(params, state)
    equal_arrays(step_arrays(resumed), expected)
    snapshot("resumed")
    report = {
        "version": 1,
        "profile": "activation",
        "precision": precision,
        "microbatch": microbatch,
        "batch_sizes": [p.stop - p.start for p in workload.parts],
        "settle_ms": settle_ms,
        "baseline_warmup": 10,
        "baseline_samples": 100,
        "resumed_step": resumed[3].step,
        "resumed_outputs_byte_equal": True,
        "preserved_input_checkpoint": True,
        "compared_outputs_and_state": list(expected),
        "steady_step_ms": {
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
        },
        "first_resumed_step_ms": resumed[-1]["step"],
        "first_resumed_scope": "One synchronized training step after idle cache release",
        "phases": phases,
        "graphs": workload.graphs,
        "provider_api": "Bundled libmlxc.dylib; mlx/c/memory.h C ABI",
        "allocator_scope": "Provider counters use bundled C ABI; Python counters use mlx.core; footprint covers the process",
        "parameter_names": list(NAMES),
        "parameter_bytes": sum(p.nbytes for p in workload.initial.values()),
        "threads": {"intra_op": 1, "inter_op": 1},
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in ("mlx", "numpy", "onnxruntime", "onnxruntime-ep-mlx", "onnxscript")
        },
    }
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="memory_lifecycle")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--microbatch", type=int, default=16384)
    parser.add_argument("--settle-ms", type=float, default=250)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                precision=args.precision,
                microbatch=args.microbatch,
                settle_ms=args.settle_ms,
            ),
            indent=2,
        )
    )
