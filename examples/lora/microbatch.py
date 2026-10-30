# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train with FP32 microbatch accumulation and measure the CPU memory boundary."""

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

import onnax
from examples.lora.accumulation import accumulate_gradients
from examples.lora.adam import adam_step
from examples.lora.model import PARAMETERS, base_model, digest, export
from examples.lora.train import close, problem, reference, reference_update, session


def gradients(executor, interface, x, y, params):
    names = ["loss", *[row["gradient"] for row in interface["parameters"]]]
    values = executor.run(names, {"features": x, "target": y, **params})
    return values[0], dict(zip(PARAMETERS, values[1:], strict=True))


def accumulated(executor, interface, x, y, params, microbatch, *, xp):
    def chunks():
        for start in range(0, len(x), microbatch):
            xb, yb = x[start : start + microbatch], y[start : start + microbatch]
            loss, grads = gradients(executor, interface, xb, yb, params)
            yield len(xb), xp.asarray(loss), {n: xp.asarray(g) for n, g in grads.items()}

    if microbatch <= 0:
        raise ValueError("Microbatch size must be positive.")
    return accumulate_gradients(chunks(), xp=xp)


def bf16_probe(directory):
    """Export actual BF16 MatMul and report CPU execution separately from checker success."""
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3, 2])
    tape = ir.tape.Tape()
    reduced = [tape.op("Cast", [v], {"to": int(ir.DataType.BFLOAT16)}) for v in (x, w)]
    output = tape.op("Cast", [tape.op("MatMul", reduced)], {"to": int(ir.DataType.FLOAT)})
    output.name = "output"
    output.type = ir.TensorType(ir.DataType.FLOAT)
    output.shape = ir.Shape([2, 2])
    graph = ir.Model(
        ir.Graph(
            [x, w],
            [output],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="bf16_matmul_probe",
        ),
        ir_version=10,
    )
    path = directory / "bf16-matmul.onnx"
    ir.save(graph, path)
    onnx.checker.check_model(str(path), full_check=True)
    try:
        onnax.jvp(graph, inputs=["x"], outputs=["output"])
    except TypeError as error:
        if "BFLOAT16" not in str(error):
            raise
        autodiff = {"supported": False, "reason": str(error)}
    else:
        raise AssertionError(
            "BF16 active-path contract changed; revalidate the precision policy."
        )
    x_value = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    w_value = np.arange(6, dtype=np.float32).reshape(3, 2) / 4
    try:
        executor = session(path)
        actual = executor.run(None, {"x": x_value, "w": w_value})[0]
    except ort.capi.onnxruntime_pybind11_state.NotImplemented as error:
        runtime = {"supported": False, "reason": str(error)}
    else:
        # Exactly BF16-representable inputs/results; this probes execution, not training accuracy.
        np.testing.assert_array_equal(actual, x_value @ w_value)
        runtime = {
            "supported": True,
            "scope": "One exactly representable MatMul; no BF16 training claim",
        }
    return {"checker": "passed", "autodiff": autodiff, "cpu_runtime": runtime}


def run(output_dir, *, microbatch=63, measure=True):
    if not isinstance(microbatch, int) or isinstance(microbatch, bool) or microbatch <= 0:
        raise ValueError("Microbatch size must be a positive integer.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    base_path = output_dir / "base.onnx"
    ir.save(base_model(w, c), base_path)
    frozen = base_path.read_bytes()
    _, _, graph, interface = export(base_path, output_dir)
    optimized = optimizer.optimize(graph.clone())
    ir.save(optimized, output_dir / "gradient.optimized.onnx")
    onnx.checker.check_model(str(output_dir / "gradient.optimized.onnx"), full_check=True)
    executor = session(output_dir / "gradient.optimized.onnx")
    xp = array_api_compat.array_namespace(*params.values())
    independent = {n: p.copy() for n, p in params.items()}
    state = ref_state = None
    history = []
    initial_validation = float(reference(vx, vy, params, w, c)[1])
    for step in range(200):
        loss, grads = accumulated(executor, interface, x, y, params, microbatch, xp=xp)
        full_loss, full_grads = gradients(executor, interface, x, y, params)
        _, expected_loss, expected_grads = reference(x, y, params, w, c)
        close(loss, full_loss)
        close(loss, expected_loss)
        for name in PARAMETERS:
            close(grads[name], full_grads[name])
            close(grads[name], expected_grads[name])
        if step == 0:
            assert np.count_nonzero(grads[PARAMETERS[0]]) == 0
            assert np.linalg.norm(grads[PARAMETERS[1]]) > 0
        if step == 1:
            assert np.linalg.norm(grads[PARAMETERS[0]]) > 0
        _, reference_loss, reference_grads = reference(x, y, independent, w, c)
        close(loss, reference_loss)
        params, state = adam_step(params, grads, state, xp=xp)
        independent, ref_state = reference_update(independent, reference_grads, ref_state)
        for name in PARAMETERS:
            close(params[name], independent[name])
            assert state.first[name].dtype == state.second[name].dtype == np.float32
        assert state.step == step + 1
        history.append(float(loss))
    validation = float(reference(vx, vy, params, w, c)[1])
    assert history[-1] < history[0] * 0.5
    assert validation < initial_validation * 0.5
    np.savez(output_dir / "adapter.npz", **params)
    with np.load(output_dir / "adapter.npz") as restored:
        prediction = session(output_dir / "forward.onnx").run(
            ["prediction"], {"features": vx, **dict(restored)}
        )[0]
        close(prediction, reference(vx, vy, params, w, c)[0])
    assert frozen == base_path.read_bytes()
    for name, expected in interface["frozen"].items():
        assert (
            digest(ir.load(output_dir / "gradient.optimized.onnx").graph.initializers[name])
            == expected
        )
    report = {
        "steps": 200,
        "effective_batch": 256,
        "microbatch": microbatch,
        "microbatch_sizes": [
            min(microbatch, len(x) - start) for start in range(0, len(x), microbatch)
        ],
        "loss_history": history,
        "validation_loss": {"initial": initial_validation, "final": validation},
        "dtype_policy": {
            "compute": "float32",
            "parameters": "float32",
            "gradients": "float32",
            "accumulation": "float32",
            "adam_state": "float32",
        },
        "bf16_probe": bf16_probe(output_dir),
    }
    report["artifacts"] = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "nodes": len(ir.load(output_dir / name).graph),
        }
        for name in ("gradient.onnx", "gradient.optimized.onnx", "bf16-matmul.onnx")
    }
    if measure:
        report["measurement"] = {
            str(size): json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "examples.lora.microbatch",
                        "--benchmark-worker",
                        str(output_dir),
                        "--microbatch",
                        str(size),
                    ],
                    text=True,
                )
            )
            for size in (16384, 512)
        }
        full, micro = (report["measurement"][str(size)] for size in (16384, 512))
        close(full["loss"], micro["loss"])
        for name in PARAMETERS:
            close(full["gradients"][name], micro["gradients"][name])
        for result in (full, micro):
            del result["gradients"]
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark(directory, microbatch):
    """Compare isolated full/microbatch processes with identical resident data."""
    interface = json.loads((directory / "interface.json").read_text())
    with np.load(directory / "adapter.npz") as data:
        params = dict(data)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16384, 16)).astype(np.float32)
    y = rng.normal(size=(16384, 8)).astype(np.float32)
    started = time.perf_counter()
    executor = session(directory / "gradient.optimized.onnx")
    creation = time.perf_counter() - started
    xp = array_api_compat.array_namespace(*params.values())
    durations = []
    # Fixed parameters isolate accumulation cost and memory, not model convergence.
    for index in range(110):
        started = time.perf_counter()
        loss, grads = accumulated(executor, interface, x, y, params, microbatch, xp=xp)
        elapsed = time.perf_counter() - started
        if index >= 10:
            durations.append(elapsed)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "loss": float(loss),
        "gradients": {n: g.tolist() for n, g in grads.items()},
        "effective_batch": len(x),
        "microbatch": microbatch,
        "resident_data_bytes": x.nbytes + y.nbytes,
        "parameter_bytes": sum(p.nbytes for p in params.values()),
        "session_creation_ms": creation * 1000,
        "median_ms": float(np.median(durations) * 1000),
        "p95_ms": float(np.percentile(durations, 95) * 1000),
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process per microbatch size; same resident CPU arrays, FP32 and model; 10 warmup/100 measured effective-batch gradients; includes CPU feed/output boundary and FP32 accumulation; excludes Adam and correctness reference; peak RSS includes imports/data/runtime/allocator, not tensor peak",
        "threads": {"intra_op": 1, "inter_op": 1},
        "platform": platform.platform(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxscript": version("onnxscript"),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "ort_optimizer": "disabled",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("microbatch_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--microbatch", type=int, default=63)
    args = parser.parse_args()
    report = (
        benchmark(args.benchmark_worker, args.microbatch)
        if args.benchmark_worker
        else run(args.output_dir, microbatch=args.microbatch)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))
