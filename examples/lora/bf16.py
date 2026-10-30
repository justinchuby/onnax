# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Validate explicit BF16 matrix compute with FP32 master state on ONNX ReferenceEvaluator."""

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
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import array_api_compat
import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass
from onnxscript import optimizer

import onnax
from examples.lora.adam import AdamState, adam_step
from examples.lora.bf16_rules import lower_matmuls
from examples.lora.microbatch import bf16_probe
from examples.lora.model import PARAMETERS, adapt, base_model, digest, with_loss
from examples.lora.train import close, problem
from examples.lora.train import reference as fp32_reference


def round_bf16(value):
    """Round finite FP32 values to nearest-even BF16 using an independent bit oracle."""
    values = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("The BF16 validation oracle requires finite inputs.")
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(
        0xFFFF0000
    )
    return rounded.view(np.float32)


def reference(x, y, params, w, c):
    """Independent forward plus analytical FP32 straight-through chain rule."""
    a, b = (params[n] for n in PARAMETERS)
    qx, qw, qa, qb, qc = (round_bf16(v) for v in (x, w, a, b, c))
    xa = round_bf16(np.dot(qx, qa))
    adapter = round_bf16(np.dot(round_bf16(xa), qb))
    z = round_bf16(np.dot(qx, qw)) + np.float32(0.25) * adapter
    hidden = np.tanh(z)
    prediction = round_bf16(np.dot(round_bf16(hidden), qc))
    residual = prediction - y
    loss = np.mean(residual * residual)
    # Stable tanh slope avoids cancellation and pins the FP32 evaluation order.
    exponential = np.exp(np.float32(-2) * np.abs(z))
    denominator = np.float32(1) + exponential
    slope = (np.float32(4) * exponential) / (denominator * denominator)
    dz = np.dot(np.float32(2 / residual.size) * residual, qc.T) * slope
    db = np.float32(0.25) * (np.dot(round_bf16(xa).T, dz))
    dxa = np.float32(0.25) * (np.dot(dz, qb.T))
    da = np.dot(qx.T, dxa)
    return prediction, loss, {PARAMETERS[0]: da, PARAMETERS[1]: db}


def reference_update(params, grads, state):
    """Independent flattened Adam, with a pinned FP32 multiply order."""
    shapes = [params[n].shape for n in PARAMETERS]
    sizes = [params[n].size for n in PARAMETERS]
    p = np.concatenate([params[n].reshape(-1) for n in PARAMETERS])
    g = np.concatenate([grads[n].reshape(-1) for n in PARAMETERS])
    step, mean, variance = (0, np.zeros_like(p), np.zeros_like(p)) if state is None else state
    step += 1
    mean = 0.9 * mean + 0.1 * g
    variance = 0.999 * variance + (0.001 * g) * g
    p = p - 0.03 * (mean / (1 - 0.9**step)) / (np.sqrt(variance / (1 - 0.999**step)) + 1e-8)
    pieces = np.split(p, np.cumsum(sizes)[:-1])
    result = {
        n: v.reshape(shape) for n, v, shape in zip(PARAMETERS, pieces, shapes, strict=True)
    }
    return result, (step, mean, variance)


def execute(executor, interface, x, y, params):
    outputs = ["prediction", "loss", *[row["gradient"] for row in interface["parameters"]]]
    values = executor.run(outputs, {"features": x, "target": y, **params})
    return values[0], values[1], dict(zip(PARAMETERS, values[2:], strict=True))


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    base_path = output_dir / "base.onnx"
    ir.save(base_model(w, c), base_path)
    frozen = base_path.read_bytes()
    source = with_loss(adapt(ir.load(base_path)))
    before = ir.to_proto(source).SerializeToString()
    mixed = lower_matmuls(source)
    mixed_before = ir.to_proto(mixed).SerializeToString()
    gradient = onnax.grad(mixed, inputs=list(PARAMETERS), output="loss")
    interface = {
        "version": 1,
        "loss": "loss",
        "loss_reduction": "mean over batch and output coordinates",
        "frozen_master": {
            n: digest(source.graph.initializers[n]) for n in ("base.W", "base.C")
        },
        "checkpoint": {
            "file": "checkpoint.npz",
            "step": "adam.step",
            "first_moment_prefix": "adam.first.",
            "second_moment_prefix": "adam.second.",
        },
        "parameters": [
            {
                "input": n,
                "gradient": value.name,
                "shape": list(p.shape),
                "master_dtype": "float32",
                "gradient_dtype": "float32",
            }
            for (n, p), value in zip(params.items(), gradient.graph.outputs[2:], strict=True)
        ],
        "compute": "BF16 matrix operands/results; FP32 nonlinearities and loss",
        "derivative_convention": "identity-through-rounding straight-through surrogate; FP32 local arithmetic",
        "runtime": "ONNX ReferenceEvaluator CPU; not a native BF16 acceleration backend",
    }
    (output_dir / "interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    artifacts = {
        "loss": mixed,
        "gradient": gradient,
        "gradient.inlined": InlinePass()(gradient.clone()).model,
    }
    # Keep frozen FP32 master initializers byte-identical; other rewrites still run.
    artifacts["gradient.optimized"] = optimizer.optimize(
        artifacts["gradient.inlined"].clone(), input_size_limit=0, output_size_limit=0
    )
    for name, graph in artifacts.items():
        path = output_dir / f"{name}.onnx"
        ir.save(graph, path)
        onnx.checker.check_model(str(path), full_check=True)
    fp32_graph = optimizer.optimize(
        onnax.grad(source, inputs=list(PARAMETERS), output="loss"),
        input_size_limit=0,
        output_size_limit=0,
    )
    ir.save(fp32_graph, output_dir / "gradient.fp32.onnx")
    onnx.checker.check_model(str(output_dir / "gradient.fp32.onnx"), full_check=True)
    executor = ReferenceEvaluator(str(output_dir / "gradient.optimized.onnx"))
    xp = array_api_compat.array_namespace(*params.values())
    independent = {n: p.copy() for n, p in params.items()}
    initial = {n: p.copy() for n, p in params.items()}
    state = reference_state = None
    history = []
    for step in range(200):
        actual = execute(executor, interface, x, y, params)
        expected = reference(x, y, params, w, c)
        for av, ev in zip(actual[:2], expected[:2], strict=True):
            close(av, ev)
        for name in PARAMETERS:
            close(actual[2][name], expected[2][name])
        if step == 0:
            assert np.count_nonzero(actual[2][PARAMETERS[0]]) == 0
            assert np.linalg.norm(actual[2][PARAMETERS[1]]) > 0
        if step == 1:
            assert np.linalg.norm(actual[2][PARAMETERS[0]]) > 0
        ref = reference(x, y, independent, w, c)
        close(actual[1], ref[1])
        params, state = adam_step(params, actual[2], state, xp=xp)
        independent, reference_state = reference_update(independent, ref[2], reference_state)
        for name in PARAMETERS:
            try:
                close(params[name], independent[name])
            except AssertionError as error:
                error.add_note(
                    f"BF16 trajectory mismatch after update {step + 1}, parameter {name}"
                )
                raise
            assert state.first[name].dtype == state.second[name].dtype == np.float32
        history.append(float(actual[1]))
    initial_validation = float(reference(vx, vy, initial, w, c)[1])
    validation = float(reference(vx, vy, params, w, c)[1])
    assert history[-1] < history[0] * 0.5
    assert validation < initial_validation * 0.5
    np.savez(output_dir / "adapter.npz", **params)
    checkpoint = {
        **params,
        "adam.step": np.array(state.step, np.int64),
        **{f"adam.first.{n}": v for n, v in state.first.items()},
        **{f"adam.second.{n}": v for n, v in state.second.items()},
    }
    np.savez(output_dir / "checkpoint.npz", **checkpoint)
    # Array/scalar conversions are an explicit external checkpoint boundary.
    with np.load(output_dir / "checkpoint.npz") as saved:
        restored_params = {n: saved[n] for n in PARAMETERS}
        restored_state = AdamState(
            int(saved["adam.step"]),
            {n: saved[f"adam.first.{n}"] for n in PARAMETERS},
            {n: saved[f"adam.second.{n}"] for n in PARAMETERS},
        )
        next_grads = execute(executor, interface, x, y, params)[2]
        updated, continued = adam_step(params, next_grads, state, xp=xp)
        resumed, resumed_state = adam_step(restored_params, next_grads, restored_state, xp=xp)
        assert continued.step == resumed_state.step == 201
        for n in PARAMETERS:
            np.testing.assert_array_equal(updated[n], resumed[n])
            np.testing.assert_array_equal(continued.first[n], resumed_state.first[n])
            np.testing.assert_array_equal(continued.second[n], resumed_state.second[n])
    with np.load(output_dir / "adapter.npz") as restored:
        for count in (1, 7, 128):
            expected = reference(vx[:count], vy[:count], params, w, c)
            for name in ("gradient", "gradient.inlined", "gradient.optimized"):
                reloaded = ReferenceEvaluator(str(output_dir / f"{name}.onnx"))
                actual = execute(reloaded, interface, vx[:count], vy[:count], dict(restored))
                close(actual[0], expected[0])
                close(actual[1], expected[1])
                for parameter in PARAMETERS:
                    close(actual[2][parameter], expected[2][parameter])
    assert before == ir.to_proto(source).SerializeToString()
    assert mixed_before == ir.to_proto(mixed).SerializeToString()
    assert frozen == base_path.read_bytes()
    for name in ("gradient", "gradient.inlined", "gradient.optimized"):
        for weight in ("base.W", "base.C"):
            assert digest(
                ir.load(output_dir / f"{name}.onnx").graph.initializers[weight]
            ) == digest(source.graph.initializers[weight])
    fp32_params = {n: p.copy() for n, p in initial.items()}
    baseline = ReferenceEvaluator(str(output_dir / "gradient.fp32.onnx"))
    fp32_state = None
    for _ in range(200):
        baseline_values = execute(baseline, interface, x, y, fp32_params)
        fp32_params, fp32_state = adam_step(fp32_params, baseline_values[2], fp32_state, xp=xp)
    np.savez(output_dir / "adapter.fp32.npz", **fp32_params)
    final_prediction = reference(vx, vy, params, w, c)[0]
    fp32_prediction = fp32_reference(vx, vy, params, w, c)[0]
    report = {
        "native_cpu_probe": bf16_probe(output_dir),
        "checkpoint_resume_verified": True,
        "master_parameter_bytes": sum(p.nbytes for p in params.values()),
        "adam_state_bytes": sum(
            v.nbytes for v in (*state.first.values(), *state.second.values())
        ),
        "steps": 200,
        "fp32_training_baseline": {
            "initial_train_loss": float(fp32_reference(x, y, initial, w, c)[1]),
            "final_train_loss": float(fp32_reference(x, y, fp32_params, w, c)[1]),
            "final_validation_loss": float(fp32_reference(vx, vy, fp32_params, w, c)[1]),
        },
        "loss_history": history,
        "validation_loss": {"initial": initial_validation, "final": validation},
        "fp32_at_same_parameters": {
            "validation_loss": float(fp32_reference(vx, vy, params, w, c)[1]),
            "prediction_max_abs_difference": float(
                np.max(np.abs(final_prediction - fp32_prediction))
            ),
        },
        "runtime": interface["runtime"],
        "dtype_policy": {
            "matmul_inputs_outputs": "bfloat16",
            "master_parameters": "float32",
            "local_derivatives": "float32",
            "accumulation": "float32",
            "adam_state": "float32",
        },
        "derivative_convention": interface["derivative_convention"],
    }
    report["artifacts"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "nodes": len(ir.load(path).graph),
            "local_functions": len(ir.load(path).functions),
        }
        for path in sorted(output_dir.glob("*.onnx"))
    }
    np.savez(output_dir / "benchmark.npz", features=x, target=y)
    report["measurement"] = {
        precision: json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "examples.lora.bf16",
                    "--benchmark-worker",
                    str(output_dir),
                    "--precision",
                    precision,
                ],
                text=True,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            )
        )
        for precision in ("bf16", "fp32")
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark(directory, precision):
    """Measure identical CPU reference backends in independent processes."""
    interface = json.loads((directory / "interface.json").read_text())
    with np.load(directory / "benchmark.npz") as data:
        x, y = data["features"], data["target"]
    with np.load(
        directory / ("adapter.npz" if precision == "bf16" else "adapter.fp32.npz")
    ) as data:
        params = dict(data)
    path = directory / (
        "gradient.optimized.onnx" if precision == "bf16" else "gradient.fp32.onnx"
    )
    start = time.perf_counter()
    payload = onnx.load(path)
    read = time.perf_counter() - start
    start = time.perf_counter()
    executor = ReferenceEvaluator(payload)
    creation = time.perf_counter() - start
    outputs = ["loss", *[row["gradient"] for row in interface["parameters"]]]
    xp = array_api_compat.array_namespace(*params.values())
    state = None
    timings = {name: [] for name in ("gradient", "adam", "step")}
    for i in range(110):
        start = time.perf_counter()
        values = executor.run(outputs, {"features": x, "target": y, **params})
        evaluated = time.perf_counter()
        params, state = adam_step(
            params, dict(zip(PARAMETERS, values[1:], strict=True)), state, xp=xp
        )
        updated = time.perf_counter()
        if i >= 10:
            timings["gradient"].append(evaluated - start)
            timings["adam"].append(updated - evaluated)
            timings["step"].append(updated - start)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    return {
        "precision": precision,
        "model_read_ms": read * 1000,
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            name: {
                "median": float(np.median(values) * 1000),
                "p95": float(np.percentile(values, 95) * 1000),
            }
            for name, values in timings.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process per precision; same CPU ONNX ReferenceEvaluator backend, batch 256; 10 warmup/100 measured updates; gradient includes feeds/outputs and reference casts/matmul; Adam uses FP32 state; RSS includes imports/runtime/allocator; not native BF16 acceleration or a GPU memory measurement",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxscript": version("onnxscript"),
        "blas": {key: blas.get(key) for key in ("name", "version")},
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("bf16_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    result = (
        benchmark(args.benchmark_worker, args.precision)
        if args.benchmark_worker
        else run(args.output_dir)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))
