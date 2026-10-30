# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train exported ONNX gradients with JAX Array API Adam and DLPack CPU buffers."""

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

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import optimizer

from examples.lora.adam import AdamState, adam_step
from examples.lora.dlpack import JaxCpuDLPack, synchronize
from examples.lora.model import PARAMETERS, base_model, digest, export
from examples.lora.train import close, problem, reference, reference_update, session


def cpu(values):
    return jax.device_put(values, jax.devices("cpu")[0])


def names(interface):
    return [interface["loss"], *[row["gradient"] for row in interface["parameters"]]]


def import_numpy(values):
    loss, *gradients = [jnp.asarray(value) for value in values]
    return loss, dict(zip(PARAMETERS, gradients, strict=True))


def host_feeds(x, y, params):
    return {n: np.asarray(v) for n, v in {"features": x, "target": y, **params}.items()}


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w, c, x, y, vx, vy, initial = problem()
    base_path = output_dir / "base.onnx"
    ir.save(base_model(w, c), base_path)
    frozen = base_path.read_bytes()
    _, _, gradient, interface = export(base_path, output_dir)
    optimized = optimizer.optimize(gradient.clone())
    path = output_dir / "gradient.optimized.onnx"
    ir.save(optimized, path)
    onnx.checker.check_model(str(path), full_check=True)
    executor = session(path)
    bound = JaxCpuDLPack(executor, interface)
    jx, jy = cpu((x, y))
    params, numpy_params = cpu((initial, initial))
    independent = {n: p.copy() for n, p in initial.items()}
    state = numpy_state = reference_state = None
    history = []
    for step in range(200):
        bound.bind(jx, jy, params)
        loss, grads = bound.run()
        host = {n: np.asarray(p) for n, p in params.items()}
        expected = reference(x, y, host, w, c)
        close(np.asarray(loss), expected[1])
        for n in PARAMETERS:
            close(np.asarray(grads[n]), expected[2][n])
        if step == 0:
            assert np.count_nonzero(np.asarray(grads[PARAMETERS[0]])) == 0
            assert np.linalg.norm(np.asarray(grads[PARAMETERS[1]])) > 0
        if step == 1:
            assert np.linalg.norm(np.asarray(grads[PARAMETERS[0]])) > 0
        normal_loss, normal_grads = import_numpy(
            executor.run(names(interface), host_feeds(jx, jy, numpy_params))
        )
        close(np.asarray(loss), np.asarray(normal_loss))
        independent_values = reference(x, y, independent, w, c)
        close(np.asarray(loss), independent_values[1])
        params, state = adam_step(params, grads, state, xp=jnp)
        numpy_params, numpy_state = adam_step(numpy_params, normal_grads, numpy_state, xp=jnp)
        independent, reference_state = reference_update(
            independent, independent_values[2], reference_state
        )
        synchronize(params, state)
        synchronize(numpy_params, numpy_state)
        for n in PARAMETERS:
            close(np.asarray(params[n]), np.asarray(numpy_params[n]))
            close(np.asarray(params[n]), independent[n])
            assert (
                params[n].dtype == state.first[n].dtype == state.second[n].dtype == jnp.float32
            )
        history.append(float(loss))
    host = {n: np.asarray(p) for n, p in params.items()}
    initial_validation = float(reference(vx, vy, initial, w, c)[1])
    validation = float(reference(vx, vy, host, w, c)[1])
    assert history[-1] < history[0] * 0.5
    assert validation < initial_validation * 0.5
    np.savez(output_dir / "adapter.npz", **host)
    np.savez(
        output_dir / "checkpoint.npz",
        **host,
        **{"adam.step": np.array(state.step, np.int64)},
        **{f"adam.first.{n}": np.asarray(v) for n, v in state.first.items()},
        **{f"adam.second.{n}": np.asarray(v) for n, v in state.second.items()},
    )
    with np.load(output_dir / "checkpoint.npz") as saved:
        restored = cpu({n: saved[n] for n in PARAMETERS})
        restored_state = AdamState(
            int(saved["adam.step"]),
            cpu({n: saved[f"adam.first.{n}"] for n in PARAMETERS}),
            cpu({n: saved[f"adam.second.{n}"] for n in PARAMETERS}),
        )
    loaded = JaxCpuDLPack(session(path), interface)
    for batch in (1, 7, 128):
        loaded.bind(*cpu((vx[:batch], vy[:batch])), restored)
        loss, grads = loaded.run()
        expected = reference(vx[:batch], vy[:batch], host, w, c)
        close(np.asarray(loss), expected[1])
        for n in PARAMETERS:
            close(np.asarray(grads[n]), expected[2][n])
    bound.bind(jx, jy, params)
    _, grads = bound.run()
    updated, next_state = adam_step(params, grads, state, xp=jnp)
    resumed, resumed_state = adam_step(restored, grads, restored_state, xp=jnp)
    synchronize(updated, next_state)
    synchronize(resumed, resumed_state)
    assert next_state.step == resumed_state.step == 201
    for n in PARAMETERS:
        np.testing.assert_array_equal(np.asarray(updated[n]), np.asarray(resumed[n]))
        np.testing.assert_array_equal(
            np.asarray(next_state.first[n]), np.asarray(resumed_state.first[n])
        )
        np.testing.assert_array_equal(
            np.asarray(next_state.second[n]), np.asarray(resumed_state.second[n])
        )
    prediction = session(output_dir / "forward.onnx").run(
        ["prediction"], {"features": vx, **host}
    )[0]
    close(prediction, reference(vx, vy, host, w, c)[0])
    assert base_path.read_bytes() == frozen
    for n, expected in interface["frozen"].items():
        assert digest(ir.load(path).graph.initializers[n]) == expected
    np.savez(output_dir / "benchmark.npz", features=x, target=y)
    report = {
        "steps": 200,
        "loss_history": history,
        "validation_loss": {"initial": initial_validation, "final": validation},
        "dtype": "float32",
        "device": "CPU",
        "checkpoint_resume_verified": True,
        "dlpack_shared_input_output_pointers": True,
        "output_lifetime": "fresh runtime-owned outputs retained by imported JAX arrays",
        "model": {"bytes": path.stat().st_size, "nodes": len(optimized.graph)},
        "parameter_bytes": sum(p.nbytes for p in host.values()),
        "optimizer_state_bytes": sum(
            v.size * v.dtype.itemsize for v in (*state.first.values(), *state.second.values())
        ),
        "input_bytes": x.nbytes + y.nbytes,
        "fetched_output_bytes": 4 + sum(p.nbytes for p in host.values()),
        "measurement": {
            mode: json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "examples.lora.jax_training",
                        "--benchmark-worker",
                        str(output_dir),
                        "--mode",
                        mode,
                    ],
                    text=True,
                )
            )
            for mode in ("dlpack", "numpy")
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark(directory, mode):
    interface = json.loads((directory / "interface.json").read_text())
    with np.load(directory / "benchmark.npz") as saved:
        x, y = cpu((saved["features"], saved["target"]))
    with np.load(directory / "adapter.npz") as saved:
        params = cpu(dict(saved))
    synchronize(params)
    started = time.perf_counter()
    executor = session(directory / "gradient.optimized.onnx")
    creation = time.perf_counter() - started
    bound = JaxCpuDLPack(executor, interface) if mode == "dlpack" else None
    output_names = names(interface)
    state = None
    timings = {n: [] for n in ("input_exchange", "execute", "output_exchange", "adam", "step")}
    for i in range(110):
        started = time.perf_counter()
        if bound is not None:
            bound.bind(x, y, params)
        else:
            feeds = host_feeds(x, y, params)
        prepared = time.perf_counter()
        if bound is not None:
            bound.execute()
        else:
            values = executor.run(output_names, feeds)
        executed = time.perf_counter()
        loss, grads = bound.import_outputs() if bound is not None else import_numpy(values)
        jax.block_until_ready([loss, *grads.values()])
        imported = time.perf_counter()
        params, state = adam_step(params, grads, state, xp=jnp)
        synchronize(params, state)
        updated = time.perf_counter()
        if i >= 10:
            timings["input_exchange"].append(prepared - started)
            timings["execute"].append(executed - prepared)
            timings["output_exchange"].append(imported - executed)
            timings["adam"].append(updated - imported)
            timings["step"].append(updated - started)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    processor = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        hardware = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        processor = hardware.stdout.strip() or processor
    return {
        "mode": mode,
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            n: {
                "median": float(np.median(v) * 1000),
                "p95": float(np.percentile(v, 95) * 1000),
            }
            for n, v in timings.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process per mode; CPUExecutionProvider/JAX CPU, batch 256, FP32, same model and external Array API Adam; 10 warmup/100 measured updates. Input exchange includes readiness and binding/feed preparation; execution includes fresh output allocation; output exchange imports and waits; Adam includes parameter and moment synchronization. Correctness-reference and checkpoint conversions are outside timing. RSS includes imports, JAX compilation/runtime and allocator.",
        "jax": jax.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": processor,
        "python": platform.python_version(),
        "onnx": onnx.__version__,
        "onnxscript": version("onnxscript"),
        "jax_cpu_devices": [str(d) for d in jax.devices("cpu")],
        "thread_environment": {
            n: os.environ.get(n)
            for n in ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES")
        },
        "ort_threads": {"intra_op": 1, "inter_op": 1},
        "ort_graph_optimizer": "disabled",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dlpack_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("dlpack", "numpy"), default="dlpack", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    result = (
        benchmark(args.benchmark_worker, args.mode)
        if args.benchmark_worker
        else run(args.output_dir)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))
