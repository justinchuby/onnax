# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run: python -m examples.lora.train --output-dir /tmp/onnax-lora."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path  # ruff: ignore[banned-api] - Example CLI paths.

import array_api_compat
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import optimizer

from examples.lora.adam import adam_step
from examples.lora.model import PARAMETERS, base_model, digest, export, validate_interface


def session(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    source = path if isinstance(path, bytes) else str(path)
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


def reference(x, target, params, w, c):
    """Independent analytic derivatives; no generated graphs or autodiff library."""
    a, b = params[PARAMETERS[0]], params[PARAMETERS[1]]
    hidden = np.tanh(x @ w + np.float32(0.25) * ((x @ a) @ b))
    prediction = hidden @ c
    residual = prediction - target
    loss = np.mean(residual * residual)
    dz = ((np.float32(2 / residual.size) * residual) @ c.T) * (1 - hidden * hidden)
    grads = {
        PARAMETERS[0]: np.float32(0.25) * (x.T @ dz @ b.T),
        PARAMETERS[1]: np.float32(0.25) * ((x @ a).T @ dz),
    }
    return prediction, loss, grads


def reference_update(params, grads, state, *, eps=1e-8):
    """Independent flattened NumPy Adam, not the Array API implementation."""
    sizes = [params[name].size for name in PARAMETERS]
    vector = np.concatenate([params[name].reshape(-1) for name in PARAMETERS])
    gradient = np.concatenate([grads[name].reshape(-1) for name in PARAMETERS])
    step, mean, variance = (
        (0, np.zeros_like(vector), np.zeros_like(vector)) if state is None else state
    )
    step += 1
    mean = 0.9 * mean + 0.1 * gradient
    variance = 0.999 * variance + 0.001 * gradient**2
    vector = vector - 0.03 * (mean / (1 - 0.9**step)) / (
        np.sqrt(variance / (1 - 0.999**step)) + eps
    )
    pieces = np.split(vector, np.cumsum(sizes)[:-1])
    return {
        name: value.reshape(params[name].shape)
        for name, value in zip(PARAMETERS, pieces, strict=True)
    }, (step, mean, variance)


def problem():
    rng = np.random.default_rng(20261029)
    w = rng.normal(0, 0.15, (16, 24)).astype(np.float32)
    c = rng.normal(0, 0.2, (24, 8)).astype(np.float32)
    teacher = {
        PARAMETERS[0]: rng.normal(0, 0.3, (16, 4)).astype(np.float32),
        PARAMETERS[1]: rng.normal(0, 0.3, (4, 24)).astype(np.float32),
    }
    x = rng.normal(size=(384, 16)).astype(np.float32)
    target = reference(x, np.zeros((384, 8), np.float32), teacher, w, c)[0]
    params = {
        PARAMETERS[0]: rng.normal(0, 0.1, (16, 4)).astype(np.float32),
        PARAMETERS[1]: np.zeros((4, 24), np.float32),
    }
    return w, c, x[:256], target[:256], x[256:], target[256:], params


def evaluate(executor, interface, x, target, params):
    required = {v.name for v in executor.get_inputs()}
    feeds = {"features": x, "target": target, **params}
    if required != feeds.keys():
        raise ValueError(
            f"Runtime feed mismatch: required={sorted(required)}, supplied={sorted(feeds)}"
        )
    if (
        x.ndim != 2
        or x.shape[1] != 16
        or x.shape[0] == 0
        or target.shape != (x.shape[0], 8)
        or x.dtype != np.float32
        or target.dtype != np.float32
    ):
        raise ValueError(
            f"Expected nonempty FLOAT features[batch,16] and target[batch,8]; "
            f"got features={x.shape}/{x.dtype}, target={target.shape}/{target.dtype}."
        )
    for row in interface["parameters"]:
        value = params[row["input"]]
        if tuple(value.shape) != tuple(row["shape"]) or value.dtype != np.float32:
            raise ValueError(f"Parameter shape/dtype mismatch for {row['input']}.")
    names = ["prediction", "loss", *[row["gradient"] for row in interface["parameters"]]]
    values = executor.run(names, feeds)
    return values[0], values[1], dict(zip(PARAMETERS, values[2:], strict=True))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def finite_differences(x, y, params, w, c):
    """Use DOUBLE analytic forward evaluation only for the independent differences."""
    x, y, w, c = (value.astype(np.float64) for value in (x, y, w, c))
    converted = {k: v.astype(np.float64) for k, v in params.items()}
    analytic = reference(x, y, converted, w, c)[2]
    for name in PARAMETERS:
        for coordinate in ((0, 0), (1, 1), (2, 2)):
            plus = {k: v.copy() for k, v in converted.items()}
            minus = {k: v.copy() for k, v in converted.items()}
            plus[name][coordinate] += 1e-4
            minus[name][coordinate] -= 1e-4
            lp = reference(x, y, plus, w, c)[1]
            lm = reference(x, y, minus, w, c)[1]
            close((lp - lm) / 2e-4, analytic[name][coordinate])


def benchmark(directory):
    """Separate-process CPU measurement; includes runtime/imports in peak RSS."""
    directory = Path(directory)
    interface = json.loads((directory / "interface.json").read_text())
    data = np.load(directory / "benchmark.npz")
    params = {name: data[name] for name in PARAMETERS}
    started = time.perf_counter()
    payload = (directory / "gradient.optimized.onnx").read_bytes()
    read_time = time.perf_counter() - started
    started = time.perf_counter()
    executor = session(payload)
    creation = time.perf_counter() - started
    xp = array_api_compat.array_namespace(*params.values())
    state = None
    times = {name: [] for name in ("gradient", "exchange", "adam", "step")}
    for index in range(110):
        start = time.perf_counter()
        # NumPy/CPU boundary: no device transfer; expose binding preparation cost.
        feeds = {k: np.asarray(v) for k, v in params.items()}
        bound = time.perf_counter()
        _, _, grads = evaluate(executor, interface, data["features"], data["target"], feeds)
        evaluated = time.perf_counter()
        grads = {k: xp.asarray(v, dtype=xp.float32) for k, v in grads.items()}
        exchanged = time.perf_counter()
        params, state = adam_step(params, grads, state, xp=xp)
        updated = time.perf_counter()
        if index >= 10:
            times["gradient"].append(evaluated - bound)
            times["exchange"].append(bound - start + exchanged - evaluated)
            times["adam"].append(updated - exchanged)
            times["step"].append(updated - start)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        hardware = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        cpu = hardware.stdout.strip() or cpu
    return {
        "cpu": cpu,
        "model_read_ms": read_time * 1000,
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            k: {
                "median": float(np.median(v) * 1000),
                "p95": float(np.percentile(v, 95) * 1000),
            }
            for k, v in times.items()
        },
        "peak_process_rss_bytes": peak_bytes,
        "scope": "CPU NumPy; 10 warmup + 100 measured updates; RSS includes imports, runtime, weights and allocator; not tensor peak or GPU memory",
        "threads": {"intra_op": 1, "inter_op": 1},
        "provider": "CPUExecutionProvider",
        "ort_optimizer": "disabled",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "onnxscript": version("onnxscript"),
        "array_api_compat": version("array-api-compat"),
    }


def run(output_dir, *, measure=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    base_path = output_dir / "base.onnx"
    ir.save(base_model(w, c), base_path)
    frozen_bytes = base_path.read_bytes()
    _forward, _, gradient, interface = export(base_path, output_dir)
    # ONNXScript only; optimize a protobuf copy, never mutate the source artifact.
    import onnx

    optimized = optimizer.optimize(ir.to_proto(gradient))
    onnx.checker.check_model(optimized, full_check=True)
    onnx.save(optimized, output_dir / "gradient.optimized.onnx")
    validate_interface(ir.load(output_dir / "gradient.optimized.onnx"), interface)
    raw = session(output_dir / "gradient.onnx")
    executor = session(output_dir / "gradient.optimized.onnx")
    xp = array_api_compat.array_namespace(*params.values())
    reference_params = {k: v.copy() for k, v in params.items()}
    state = reference_state = None
    initial_train = float(reference(x, y, params, w, c)[1])
    initial_validation = float(reference(vx, vy, params, w, c)[1])
    history = []
    base_prediction = session(base_path).run(["prediction"], {"features": x})[0]
    for step in range(200):
        actual = evaluate(executor, interface, x, y, params)
        independent = reference(x, y, params, w, c)
        for av, ev in zip(actual[:2], independent[:2], strict=True):
            close(av, ev)
        for name in PARAMETERS:
            close(actual[2][name], independent[2][name])
        if step == 0:
            close(actual[0], base_prediction)
            assert np.count_nonzero(actual[2][PARAMETERS[0]]) == 0
            assert np.linalg.norm(actual[2][PARAMETERS[1]]) > 0
        if step == 1:
            assert np.linalg.norm(actual[2][PARAMETERS[0]]) > 0
            finite_differences(x, y, params, w, c)
        # Independently evolving reference trajectory, with the same external update.
        ref_prediction, ref_loss, ref_grads = reference(x, y, reference_params, w, c)
        close(actual[0], ref_prediction)
        close(actual[1], ref_loss)
        params, state = adam_step(params, actual[2], state, xp=xp)
        reference_params, reference_state = reference_update(
            reference_params, ref_grads, reference_state
        )
        for name in PARAMETERS:
            close(params[name], reference_params[name])
        history.append(float(actual[1]))
    final_train = float(reference(x, y, params, w, c)[1])
    final_validation = float(reference(vx, vy, params, w, c)[1])
    assert final_train <= initial_train * 0.5
    assert final_validation <= initial_validation * 0.5
    # Compare raw/optimized modules and exercise dynamic batches after reload.
    reloaded = session(output_dir / "gradient.optimized.onnx")
    for batch in (1, 7, 128):
        raw_values = evaluate(raw, interface, vx[:batch], vy[:batch], params)
        loaded_values = evaluate(reloaded, interface, vx[:batch], vy[:batch], params)
        expected = reference(vx[:batch], vy[:batch], params, w, c)
        for values in (raw_values, loaded_values):
            close(values[0], expected[0])
            close(values[1], expected[1])
            for name in PARAMETERS:
                close(values[2][name], expected[2][name])
    np.savez(output_dir / "adapter.npz", **params)
    restored = dict(np.load(output_dir / "adapter.npz"))
    prediction = session(output_dir / "forward.onnx").run(
        ["prediction"], {"features": vx, **restored}
    )[0]
    close(prediction, reference(vx, vy, params, w, c)[0])
    assert base_path.read_bytes() == frozen_bytes
    for name in ("base.W", "base.C"):
        for artifact in ("forward", "loss", "gradient", "gradient.optimized"):
            assert (
                digest(ir.load(output_dir / f"{artifact}.onnx").graph.initializers[name])
                == interface["frozen"][name]
            )
    np.savez(output_dir / "benchmark.npz", features=x, target=y, **params)
    report = {
        "steps": 200,
        "train_loss": {"initial": initial_train, "final": final_train},
        "validation_loss": {"initial": initial_validation, "final": final_validation},
        "loss_history": history,
        "parameter_bytes": sum(v.nbytes for v in params.values()),
        "artifacts": {
            name: {
                "bytes": (output_dir / f"{name}.onnx").stat().st_size,
                "nodes": len(ir.load(output_dir / f"{name}.onnx").graph),
            }
            for name in ("forward", "loss", "gradient", "gradient.optimized")
        },
    }
    if measure:
        result = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "examples.lora.train",
                "--benchmark-worker",
                str(output_dir),
            ],
            text=True,
        )
        report["measurement"] = json.loads(result)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("lora_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.benchmark_worker:
        print(json.dumps(benchmark(args.benchmark_worker)))
    else:
        report = run(args.output_dir)
        print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
