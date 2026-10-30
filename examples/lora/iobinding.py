# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train LoRA with reusable CPU gradient buffers: python -m examples.lora.iobinding."""

from __future__ import annotations

import argparse
import json
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
import onnxruntime as ort
from onnxscript import optimizer

from examples.lora.adam import adam_step
from examples.lora.binding import CpuGradientBinding
from examples.lora.microbatch import gradients
from examples.lora.model import PARAMETERS, base_model, digest, export
from examples.lora.train import close, problem, reference, reference_update, session


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    base_path = output_dir / "base.onnx"
    ir.save(base_model(w, c), base_path)
    frozen = base_path.read_bytes()
    _, _, gradient, interface = export(base_path, output_dir)
    optimized = optimizer.optimize(gradient.clone())
    path = output_dir / "gradient.optimized.onnx"
    ir.save(optimized, path)
    onnx.checker.check_model(str(path), full_check=True)
    executor = session(path)
    bound = CpuGradientBinding(executor, interface)
    output_pointers = {n: v.ctypes.data for n, v in bound.output_arrays.items()}
    standard = {n: p.copy() for n, p in params.items()}
    independent = {n: p.copy() for n, p in params.items()}
    xp = array_api_compat.array_namespace(*params.values())
    state = standard_state = reference_state = None
    initial_validation = float(reference(vx, vy, params, w, c)[1])
    history = []
    for step in range(200):
        bound.bind(x, y, params)
        loss, grads = bound.run()
        same_point = reference(x, y, params, w, c)
        close(loss, same_point[1])
        for name in PARAMETERS:
            close(grads[name], same_point[2][name])
        normal_loss, normal_grads = gradients(executor, interface, x, y, standard)
        close(loss, normal_loss)
        for name in PARAMETERS:
            close(grads[name], normal_grads[name])
        if step == 0:
            assert np.count_nonzero(grads[PARAMETERS[0]]) == 0
            assert np.linalg.norm(grads[PARAMETERS[1]]) > 0
        if step == 1:
            assert np.linalg.norm(grads[PARAMETERS[0]]) > 0
        ref = reference(x, y, independent, w, c)
        close(loss, ref[1])
        params, state = adam_step(params, grads, state, xp=xp)
        standard, standard_state = adam_step(standard, normal_grads, standard_state, xp=xp)
        independent, reference_state = reference_update(independent, ref[2], reference_state)
        for name in PARAMETERS:
            close(params[name], standard[name])
            close(params[name], independent[name])
        assert {n: v.ctypes.data for n, v in bound.output_arrays.items()} == output_pointers
        history.append(float(loss))
    validation = float(reference(vx, vy, params, w, c)[1])
    assert history[-1] < history[0] * 0.5
    assert validation < initial_validation * 0.5
    np.savez(output_dir / "adapter.npz", **params)
    with np.load(output_dir / "adapter.npz") as saved:
        restored = dict(saved)
        reloaded = CpuGradientBinding(session(path), interface)
        for count in (1, 7, 128):
            reloaded.bind(vx[:count], vy[:count], restored)
            loss, grads = reloaded.run()
            expected = reference(vx[:count], vy[:count], restored, w, c)
            close(loss, expected[1])
            for n in PARAMETERS:
                close(grads[n], expected[2][n])
        prediction = session(output_dir / "forward.onnx").run(
            ["prediction"], {"features": vx, **restored}
        )[0]
        close(prediction, reference(vx, vy, params, w, c)[0])
    assert base_path.read_bytes() == frozen
    for name, expected in interface["frozen"].items():
        assert digest(ir.load(path).graph.initializers[name]) == expected
    np.savez(output_dir / "benchmark.npz", features=x, target=y)
    report = {
        "steps": 200,
        "parameter_bytes": sum(p.nbytes for p in params.values()),
        "input_bytes": x.nbytes + y.nbytes,
        "loss_history": history,
        "validation_loss": {"initial": initial_validation, "final": validation},
        "output_buffer_bytes": sum(v.nbytes for v in bound.output_arrays.values()),
        "stable_output_addresses": True,
        "model": {"bytes": path.stat().st_size, "nodes": len(optimized.graph)},
        "measurement": {
            mode: json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "examples.lora.iobinding",
                        "--benchmark-worker",
                        str(output_dir),
                        "--mode",
                        mode,
                    ],
                    text=True,
                )
            )
            for mode in ("run", "binding")
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark(directory, mode):
    """Measure matching CPU loss/gradient outputs and complete external Adam steps."""
    interface = json.loads((directory / "interface.json").read_text())
    with np.load(directory / "benchmark.npz") as saved:
        x, y = saved["features"], saved["target"]
    with np.load(directory / "adapter.npz") as saved:
        params = dict(saved)
    started = time.perf_counter()
    executor = session(directory / "gradient.optimized.onnx")
    creation = time.perf_counter() - started
    names = [interface["loss"], *[row["gradient"] for row in interface["parameters"]]]
    started = time.perf_counter()
    bound = CpuGradientBinding(executor, interface) if mode == "binding" else None
    binding_setup = time.perf_counter() - started if bound is not None else None
    xp = array_api_compat.array_namespace(*params.values())
    state = None
    timings = {n: [] for n in ("bind_or_feed", "execute", "adam", "step")}
    for index in range(110):
        started = time.perf_counter()
        if bound is not None:
            bound.bind(x, y, params)
        else:
            feeds = {"features": x, "target": y, **params}
        prepared = time.perf_counter()
        if bound is not None:
            _loss, grads = bound.run()
        else:
            values = executor.run(names, feeds)
            grads = dict(zip(PARAMETERS, values[1:], strict=True))
        executed = time.perf_counter()
        params, state = adam_step(params, grads, state, xp=xp)
        updated = time.perf_counter()
        if index >= 10:
            timings["bind_or_feed"].append(prepared - started)
            timings["execute"].append(executed - prepared)
            timings["adam"].append(updated - executed)
            timings["step"].append(updated - started)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        hardware = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        cpu = hardware.stdout.strip() or cpu
    return {
        "mode": mode,
        "session_creation_ms": creation * 1000,
        "output_binding_setup_ms": binding_setup * 1000 if binding_setup is not None else None,
        "timings_ms": {
            n: {
                "median": float(np.median(v) * 1000),
                "p95": float(np.percentile(v, 95) * 1000),
            }
            for n, v in timings.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process per mode; same FP32 CPU model, batch 256, parameters, loss/gradient fetches and Adam; 10 warmup/100 measured updates. Binding preparation includes metadata/contiguity/overlap checks and changed-input OrtValue creation; normal feed preparation builds a dictionary. Execute includes Python result handling. RSS includes imports, runtime and allocator.",
        "platform": platform.platform(),
        "cpu": cpu,
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "onnxscript": version("onnxscript"),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "threads": {"intra_op": 1, "inter_op": 1},
        "ort_graph_optimizer": "disabled",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("iobinding_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("run", "binding"), default="binding", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    result = (
        benchmark(args.benchmark_worker, args.mode)
        if args.benchmark_worker
        else run(args.output_dir)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))
