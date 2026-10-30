# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train a classifier with sample-weighted FP32 gradient accumulation on Metal."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

from examples.lora.accumulation import accumulate_gradients
from examples.lora.mlx_adam import CompiledAdam
from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import host
from examples.lora.validation import discrepancy
from examples.mlp import NAMES, reference_update
from examples.mlp_bf16 import graph, reference
from examples.mlp_device import PROFILES, problem
from examples.mlx_binding import MlxBinding, synchronize
from examples.process_memory import physical_footprint


def slices(rows, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Expected a positive integer microbatch size")
    return [slice(start, min(start + size, rows)) for start in range(0, rows, size)]


class Workload:
    """Reuse static sessions per batch size with shared Metal inputs/outputs."""

    def __init__(self, directory, *, profile, precision, microbatch, reload=False):
        import mlx.core as mx

        self.xp = mx
        self.profile, self.precision = profile, precision
        self.x, self.y, self.vx, self.vy, self.initial = problem(profile)
        width, hidden, classes = PROFILES[profile][:3]
        self.parts = slices(len(self.x), microbatch)
        self.bindings, self.mapping = {}, None
        self.data = []
        self.session_ms = 0.0
        self.graphs = {}
        os.makedirs(directory, exist_ok=True)
        for part in self.parts:
            count = part.stop - part.start
            if count not in self.bindings:
                path = os.path.join(directory, f"gradient-{count}.onnx")
                if reload:
                    model = ir.load(path)
                else:
                    raw = graph(
                        precision=precision,
                        batch=count,
                        input_width=width,
                        hidden_width=hidden,
                        classes=classes,
                        optimize=False,
                    )
                    raw.graph.outputs.remove(raw.graph.outputs[0])
                    model = optimizer.optimize(raw)
                    ir.save(model, path)
                onnx.checker.check_model(ir.to_proto(model), full_check=True)
                self.graphs[count] = {
                    "nodes": len(model.graph),
                    "bytes": os.path.getsize(path),
                }
                mapping = dict(
                    zip(NAMES, [v.name for v in model.graph.outputs[1:]], strict=True)
                )
                if self.mapping is not None and self.mapping != mapping:
                    raise ValueError("Batch-specific gradient names changed")
                self.mapping = mapping
                started = time.perf_counter()
                self.bindings[count] = MlxBinding(session(model))
                self.session_ms += (time.perf_counter() - started) * 1000
            self.data.append(
                {"features": mx.array(self.x[part]), "targets": mx.array(self.y[part])}
            )
        for data in self.data:
            synchronize(data)
        self.update = CompiledAdam(
            lr=0.03 if profile == "tiny" else 0.01, eps=1e-8 if profile == "tiny" else 1e-6
        )

    def parameters(self):
        params = {n: self.xp.array(p) for n, p in self.initial.items()}
        synchronize(params)
        return params

    def gradients(self, params):
        timings = {"execute_exchange": 0.0, "accumulate_sync": 0.0}

        def chunks():
            for part, data in zip(self.parts, self.data, strict=True):
                started = time.perf_counter()
                count = part.stop - part.start
                binding = self.bindings[count]
                binding.bind({**data, **params})
                outputs = binding.run()
                timings["execute_exchange"] += (time.perf_counter() - started) * 1000
                yield count, outputs["loss"], {n: outputs[o] for n, o in self.mapping.items()}

        def evaluate(*arrays):
            started = time.perf_counter()
            self.xp.eval(*arrays)
            timings["accumulate_sync"] += (time.perf_counter() - started) * 1000

        if len(self.parts) == 1:
            _, loss, grads = next(chunks())
        else:
            loss, grads = accumulate_gradients(chunks(), xp=self.xp, evaluate=evaluate)
            evaluate(loss, *grads.values())
        return loss, grads, timings

    def step(self, params, state):
        started = time.perf_counter()
        loss, grads, timings = self.gradients(params)
        before_update = time.perf_counter()
        params, state = self.update(params, grads, state)
        synchronize(params, state)
        finished = time.perf_counter()
        timings.update(
            adam_sync=(finished - before_update) * 1000, step=(finished - started) * 1000
        )
        return loss, grads, params, state, timings


def benchmark(directory, *, profile, precision, microbatch):
    workload = Workload(directory, profile=profile, precision=precision, microbatch=microbatch)
    params, state = workload.parameters(), None
    samples = []
    first_step = None
    for i in range(110):
        loss, grads, params, state, timings = workload.step(params, state)
        del loss, grads
        if i == 0:
            first_step = timings
        if i >= 10:
            samples.append(timings)
    return {
        "profile": profile,
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in ("numpy", "mlx", "onnxruntime", "onnxruntime-ep-mlx", "onnxscript")
        },
        "threads": {"intra_op": 1, "inter_op": 1},
        "parameter_bytes": sum(p.nbytes for p in workload.initial.values()),
        "precision": precision,
        "microbatch": microbatch,
        "effective_batch": len(workload.x),
        "batch_sizes": [p.stop - p.start for p in workload.parts],
        "single_hidden_tensor_bytes": max(p.stop - p.start for p in workload.parts)
        * PROFILES[profile][1]
        * 4,
        "hidden_tensor_scope": "Size of one FP32 batch-by-hidden tensor; analytical size, not measured peak",
        "graphs": workload.graphs,
        "session_creation_ms": workload.session_ms,
        "first_step_ms": first_step,
        "warmup": 10,
        "samples": 100,
        "timings_ms": {
            n: {
                "median": float(np.median([s[n] for s in samples])),
                "p95": float(np.percentile([s[n] for s in samples], 95)),
            }
            for n in samples[0]
        },
        "physical_footprint": physical_footprint(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Independent macOS process including Python, graph construction and runtime",
    }


def run(directory, *, profile="medium", precision="fp32", microbatch=300, measure=True):
    workload = Workload(directory, profile=profile, precision=precision, microbatch=microbatch)
    full = Workload(
        os.path.join(directory, "full"),
        profile=profile,
        precision=precision,
        microbatch=len(workload.x),
    )
    params, state = workload.parameters(), None
    oracle, oracle_state = {n: p.copy() for n, p in workload.initial.items()}, None
    update_state = None
    history, errors = [], {}

    def compare(label, actual, expected, step):
        difference = discrepancy(actual, expected)
        if difference["mismatched"]:
            errors.setdefault(label, {"first_step": step, **difference})

    for step in range(150):
        before = {n: host(p).copy() for n, p in params.items()}
        full_loss, full_grads, _ = full.gradients(params)
        old_params = params
        loss, grads, params, state, _ = workload.step(params, state)
        ref_logits, ref_loss, ref_grads = reference(
            workload.x, workload.y, before, precision=precision
        )
        del ref_logits
        compare("full.loss", host(loss), host(full_loss), step)
        compare("reference.loss", host(loss), ref_loss, step)
        for n in NAMES:
            compare("full." + n, host(grads[n]), host(full_grads[n]), step)
            compare("reference." + n, host(grads[n]), ref_grads[n], step)
        expected, update_state = reference_update(
            before,
            {n: host(g) for n, g in grads.items()},
            update_state,
            lr=workload.update.lr,
            eps=workload.update.eps,
        )
        _, oracle_loss, oracle_grads = reference(
            workload.x, workload.y, oracle, precision=precision
        )
        compare("trajectory.loss", host(loss), oracle_loss, step)
        oracle, oracle_state = reference_update(
            oracle, oracle_grads, oracle_state, lr=workload.update.lr, eps=workload.update.eps
        )
        for n in NAMES:
            compare("trajectory." + n, host(params[n]), oracle[n], step)
            np.testing.assert_allclose(host(params[n]), expected[n], rtol=1e-5, atol=1e-8)
            np.testing.assert_array_equal(host(old_params[n]), before[n])
        offset = 0
        for n in NAMES:
            size = before[n].size
            for group, flattened in (
                (state.first, update_state[1]),
                (state.second, update_state[2]),
            ):
                np.testing.assert_allclose(
                    host(group[n]),
                    flattened[offset : offset + size].reshape(before[n].shape),
                    rtol=1e-5,
                    atol=1e-15,
                )
            offset += size
        arrays = (
            list(params.values())
            + list(state.first.values())
            + list(state.second.values())
            + list(grads.values())
            + [loss]
        )
        assert state.step == step + 1
        assert all(
            a.dtype == workload.xp.float32 and a.__dlpack_device__() == (8, 0) for a in arrays
        )
        history.append(float(host(loss)))
    final = {n: host(p).copy() for n, p in params.items()}
    initial_train = reference(workload.x, workload.y, workload.initial, precision=precision)[1]
    final_train = reference(workload.x, workload.y, final, precision=precision)[1]
    initial_val = reference(workload.vx, workload.vy, workload.initial, precision=precision)[1]
    final_val = reference(workload.vx, workload.vy, final, precision=precision)[1]
    assert final_train < initial_train * 0.5 and final_val < initial_val * 0.5
    checkpoint = {**final, "adam.step": np.array(state.step, dtype=np.int64)}
    for group in ("first", "second"):
        checkpoint.update(
            {f"adam.{group}.{n}": host(a).copy() for n, a in getattr(state, group).items()}
        )
    np.savez(os.path.join(directory, "checkpoint.npz"), **checkpoint)
    # Reload graph sessions and every master/moment before update 151.
    from examples.lora.adam import AdamState

    restored = Workload(
        directory,
        profile=profile,
        precision=precision,
        microbatch=microbatch,
        reload=True,
    )
    with np.load(os.path.join(directory, "checkpoint.npz")) as saved:
        restored_params = {n: workload.xp.array(saved[n]) for n in NAMES}
        restored_state = AdamState(
            int(saved["adam.step"]),
            {n: workload.xp.array(saved[f"adam.first.{n}"]) for n in NAMES},
            {n: workload.xp.array(saved[f"adam.second.{n}"]) for n in NAMES},
        )
    a = workload.step(params, state)
    b = restored.step(restored_params, restored_state)
    assert a[3].step == b[3].step == 151
    np.testing.assert_array_equal(host(a[0]), host(b[0]))
    for group_a, group_b in [
        (a[1], b[1]),
        (a[2], b[2]),
        (a[3].first, b[3].first),
        (a[3].second, b[3].second),
    ]:
        for n in NAMES:
            np.testing.assert_array_equal(host(group_a[n]), host(group_b[n]))
    width, hidden, classes = PROFILES[profile][:3]
    validation = graph(
        precision=precision,
        batch=len(workload.vx),
        input_width=width,
        hidden_width=hidden,
        classes=classes,
        with_gradients=False,
    )
    validation_path = os.path.join(directory, "validation.onnx")
    ir.save(validation, validation_path)
    onnx.checker.check_model(ir.to_proto(validation), full_check=True)
    validation_binding = MlxBinding(session(ir.load(validation_path)))
    validation_binding.bind(
        {
            "features": workload.xp.array(workload.vx),
            "targets": workload.xp.array(workload.vy),
            **params,
        }
    )
    native_validation = validation_binding.run()
    native_validation_loss = float(host(native_validation["loss"]))
    assert native_validation_loss < initial_val * 0.5
    assert float(host(a[0])) < initial_train * 0.5
    compare("final_validation.loss", host(native_validation["loss"]), final_val, 150)
    report = {
        "profile": profile,
        "precision": precision,
        "microbatch": microbatch,
        "batch_sizes": [p.stop - p.start for p in workload.parts],
        "steps": 150,
        "rtol": 1e-4,
        "atol": 1e-6,
        "errors": errors,
        "history": history,
        "initial_train_loss": float(initial_train),
        "final_train_loss": float(final_train),
        "initial_validation_loss": float(initial_val),
        "final_validation_loss": float(final_val),
        "native_final_validation_loss": native_validation_loss,
        "native_final_train_loss": float(host(a[0])),
        "checkpoint_replay": True,
        "masters_moments_on_metal": True,
        "shared_gradient_update_acceptance": True,
        "pointer_identity_checked": True,
        "parameter_bytes": sum(p.nbytes for p in final.values()),
        "graphs": workload.graphs,
        "threads": {"intra_op": 1, "inter_op": 1},
        "same_parameter_acceptance": not any(
            n.startswith(("full.", "reference.")) for n in errors
        ),
        "independent_trajectory_acceptance": not any(
            n.startswith("trajectory.") for n in errors
        ),
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in ("numpy", "mlx", "onnxruntime", "onnxruntime-ep-mlx", "onnxscript")
        },
    }
    if measure:
        for name, size in [("full", len(workload.x)), ("microbatch", microbatch)]:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "examples.mlp_microbatch",
                    "--worker",
                    "--output-dir",
                    os.path.join(directory, "benchmark-" + name),
                    "--profile",
                    profile,
                    "--precision",
                    precision,
                    "--microbatch",
                    str(size),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report[name + "_measurement"] = json.loads(completed.stdout)
    with open(os.path.join(directory, "interface.json"), "w") as stream:
        json.dump(
            {
                "version": 1,
                "parameters": [
                    {
                        "input": n,
                        "gradient": workload.mapping[n],
                        "shape": list(p.shape),
                        "dtype": "float32",
                    }
                    for n, p in workload.initial.items()
                ],
                "loss": "loss",
                "reduction": "mean over samples; sum over classes",
                "batch_sizes": sorted(workload.bindings),
                "precision": precision,
            },
            stream,
            indent=2,
        )
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="microbatch_models")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="medium")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--microbatch", type=int, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-only",
        dest="worker",
        action="store_true",
        help="Measure 10 warmup and 100 synchronized training steps",
    )
    args = parser.parse_args()
    function = benchmark if args.worker else run
    print(
        json.dumps(
            function(
                args.output_dir,
                profile=args.profile,
                precision=args.precision,
                microbatch=args.microbatch,
            ),
            indent=2,
        )
    )
