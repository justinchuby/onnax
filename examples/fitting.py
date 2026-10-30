# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Fit an exponential decay with exported matrix-free Gauss-Newton products."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import optimizer

import onnax
from examples.lora.train import session


def model():
    """Return residuals and half squared error; the observation count is dynamic."""
    p = ir.val("parameters", dtype=ir.DataType.DOUBLE, shape=[3])
    t = ir.val("time", dtype=ir.DataType.DOUBLE, shape=["observations"])
    y = ir.val("observed", dtype=ir.DataType.DOUBLE, shape=["observations"])
    tape = ir.tape.Tape()
    terms = [
        tape.op(
            "Gather",
            [p, tape.initializer(ir.tensor(np.array(i, np.int64), name=f"index_{i}"))],
            {"axis": 0},
        )
        for i in range(3)
    ]
    decay = tape.op("Exp", [tape.op("Neg", [tape.op("Mul", [terms[1], t])])])
    prediction = tape.op("Add", [tape.op("Mul", [terms[0], decay]), terms[2]])
    residual = tape.op("Sub", [prediction, y])
    residual.name = "residual"
    residual.type = ir.TensorType(ir.DataType.DOUBLE)
    residual.shape = ir.Shape(["observations"])
    half = tape.initializer(ir.tensor(np.array(0.5, np.float64), name="half"))
    loss = tape.op(
        "Mul",
        [half, tape.op("ReduceSum", [tape.op("Mul", [residual, residual])], {"keepdims": 0})],
    )
    loss.name = "loss"
    loss.type = ir.TensorType(ir.DataType.DOUBLE)
    loss.shape = ir.Shape([])
    return ir.Model(
        ir.Graph(
            [p, t, y],
            [residual, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="decay_fit",
        ),
        ir_version=10,
    )


def reference(p, t, y):
    """Independent analytic residual, Jacobian and exact loss Hessian."""
    a, rate, offset = p
    exponential = np.exp(-rate * t)
    residual = a * exponential + offset - y
    jacobian = np.column_stack([exponential, -a * t * exponential, np.ones_like(t)])
    correction = np.zeros((3, 3))
    correction[0, 1] = correction[1, 0] = np.sum(-residual * t * exponential)
    correction[1, 1] = np.sum(residual * a * t * t * exponential)
    return residual, jacobian, jacobian.T @ jacobian + correction


def cg(product, rhs, *, tolerance=1e-11, max_iterations=12):
    """Solve an SPD system using only vector products; reject invalid curvature."""
    x = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    squared = residual @ residual
    if np.sqrt(squared) <= tolerance:
        return x, 0
    for iteration in range(max_iterations):
        action = product(direction)
        curvature = direction @ action
        if not np.isfinite(curvature) or curvature <= 0:
            raise ValueError("CG requires finite positive curvature")
        alpha = squared / curvature
        x = x + alpha * direction
        residual = residual - alpha * action
        updated = residual @ residual
        if np.sqrt(updated) <= tolerance:
            return x, iteration + 1
        direction = residual + (updated / squared) * direction
        squared = updated
    raise RuntimeError("CG did not reach the requested residual tolerance")


def solve(residual, transpose, product, initial, *, iterations=12, damping=1e-3):
    """Run external damped Gauss-Newton with bounded Armijo backtracking."""
    p = initial.copy()
    history = []
    for _ in range(iterations):
        r = residual(p)
        loss = 0.5 * (r @ r)
        g = transpose(p, r)
        history.append(float(loss))
        if np.linalg.norm(g) < 1e-9:
            break
        delta, _ = cg(lambda v, p=p: product(p, v) + damping * v, -g)
        for backtrack in range(20):
            scale = 0.5**backtrack
            candidate = p + scale * delta
            trial = residual(candidate)
            if 0.5 * (trial @ trial) <= loss + 1e-4 * scale * (g @ delta):
                p = candidate
                break
        else:
            raise RuntimeError("Gauss-Newton line search failed")
    final = residual(p)
    history.append(float(0.5 * (final @ final)))
    return p, history


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    source = model()
    before = ir.to_proto(source).SerializeToString()
    graphs = {
        "residual": source,
        "jvp": onnax.jvp(source, inputs=["parameters"], outputs=["residual"]),
        "vjp": onnax.vjp(source, inputs=["parameters"], outputs=["residual"]),
        "hvp": onnax.jvp(
            onnax.grad(source, inputs=["parameters"], output="loss"), inputs=["parameters"]
        ),
    }
    graphs["batched_jvp"] = onnax.vmap(graphs["jvp"], in_axes=(None, None, None, 0))
    executors = {}
    for name, graph in graphs.items():
        raw = output_dir / f"{name}.onnx"
        ir.save(graph, raw)
        onnx.checker.check_model(str(raw), full_check=True)
        optimized = optimizer.optimize(graph.clone())
        path = output_dir / f"{name}.optimized.onnx"
        ir.save(optimized, path)
        onnx.checker.check_model(str(path), full_check=True)
        executors[name] = session(path)
    interface = {
        "version": 1,
        "dtype": "float64",
        "parameters": ["amplitude", "decay_rate", "offset"],
        "graphs": {
            name: {
                "file": f"{name}.optimized.onnx",
                "inputs": [value.name for value in graph.graph.inputs],
                "outputs": [value.name for value in graph.graph.outputs],
            }
            for name, graph in graphs.items()
        },
    }
    (output_dir / "interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    rng = np.random.default_rng(42)
    t = np.linspace(0, 3, 128)
    truth = np.array([2.0, 1.3, 0.4])
    y = truth[0] * np.exp(-truth[1] * t) + truth[2] + rng.normal(0, 0.005, t.shape)
    initial = np.array([1.4, 0.8, 0.1])
    base = {"time": t, "observed": y}

    def execute(name, p, seed=None):
        executor = executors[name]
        feeds = {**base, "parameters": p}
        if seed is not None:
            feeds[executor.get_inputs()[-1].name] = seed
        return executor.run(None, feeds)[-1 if seed is not None else 0]

    def transpose(p, v):
        return execute("vjp", p, v)

    def product(p, v):
        return transpose(p, execute("jvp", p, v))

    # Dense matrices exist only in this independent small validation oracle.
    for count in (1, 7, 128):
        feeds = {"parameters": initial, "time": t[:count], "observed": y[:count]}
        r, jacobian, hessian = reference(initial, t[:count], y[:count])
        direction = rng.normal(size=3)
        seed = rng.normal(size=count)
        for name, vector, expected in (
            ("jvp", direction, jacobian @ direction),
            ("vjp", seed, jacobian.T @ seed),
            ("hvp", direction, hessian @ direction),
        ):
            for suffix in ("", ".optimized"):
                executor = session(output_dir / f"{name}{suffix}.onnx")
                actual = executor.run(None, {**feeds, executor.get_inputs()[-1].name: vector})[
                    -1
                ]
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            execute("residual", initial)[:count], r, rtol=1e-9, atol=1e-10
        )
    directions = rng.normal(size=(5, 3))
    batched = executors["batched_jvp"]
    actual = batched.run(
        None, {**base, "parameters": initial, batched.get_inputs()[-1].name: directions}
    )[-1]
    jacobian = reference(initial, t, y)[1]
    np.testing.assert_allclose(actual, directions @ jacobian.T, rtol=1e-9, atol=1e-10)
    direction = directions[0]
    cotangent = rng.normal(size=len(t))
    np.testing.assert_allclose(
        execute("jvp", initial, direction) @ cotangent,
        direction @ transpose(initial, cotangent),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        product(initial, direction), jacobian.T @ (jacobian @ direction), rtol=1e-9, atol=1e-10
    )
    # An exact HVP includes residual curvature; Gauss-Newton deliberately omits it.
    assert (
        np.linalg.norm(execute("hvp", initial, direction) - product(initial, direction)) > 0.1
    )
    parameters, history = solve(lambda p: execute("residual", p), transpose, product, initial)
    expected, expected_history = solve(
        lambda p: reference(p, t, y)[0],
        lambda p, v: reference(p, t, y)[1].T @ v,
        lambda p, v: reference(p, t, y)[1].T @ (reference(p, t, y)[1] @ v),
        initial,
    )
    np.testing.assert_allclose(parameters, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(history, expected_history, rtol=1e-9, atol=1e-10)
    assert history[-1] < history[0] * 0.01
    assert np.linalg.norm(transpose(parameters, execute("residual", parameters))) < 1e-8
    assert np.linalg.norm(parameters - truth) < 0.02
    np.savez(output_dir / "fit.npz", parameters=parameters, time=t, observed=y)
    with np.load(output_dir / "fit.npz") as restored:
        reloaded = session(output_dir / "residual.optimized.onnx")
        actual = reloaded.run(
            None,
            {
                "parameters": restored["parameters"],
                "time": restored["time"],
                "observed": restored["observed"],
            },
        )[0]
        np.testing.assert_allclose(
            actual, reference(parameters, t, y)[0], rtol=1e-9, atol=1e-10
        )
    assert before == ir.to_proto(source).SerializeToString()
    report = {
        "parameters": parameters.tolist(),
        "truth": truth.tolist(),
        "loss_history": history,
        "observations": len(t),
        "dtype": "float64",
        "method": "damped Gauss-Newton with matrix-free CG",
        "ort_graph_optimization": "disabled",
    }
    report["artifacts"] = {
        p.name: {"bytes": p.stat().st_size, "nodes": len(ir.load(p).graph)}
        for p in sorted(output_dir.glob("*.onnx"))
    }
    report["measurement"] = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "examples.fitting", "--benchmark-worker", str(output_dir)],
            text=True,
        )
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark(directory):
    """Measure a fixed-point product and complete solve in an isolated CPU process."""
    with np.load(directory / "fit.npz") as data:
        base = {"time": data["time"], "observed": data["observed"]}
    start = time.perf_counter()
    executors = {
        name: session(directory / f"{name}.optimized.onnx")
        for name in ("residual", "jvp", "vjp")
    }
    creation = time.perf_counter() - start

    def execute(name, p, seed=None):
        executor = executors[name]
        feeds = {**base, "parameters": p}
        if seed is not None:
            feeds[executor.get_inputs()[-1].name] = seed
        return executor.run(
            [executor.get_outputs()[-1 if seed is not None else 0].name], feeds
        )[0]

    def product(p, v):
        return execute("vjp", p, execute("jvp", p, v))

    initial = np.array([1.4, 0.8, 0.1])
    direction = np.array([0.1, -0.2, 0.3])
    timings = {name: [] for name in ("gauss_newton_product", "complete_solve")}
    for i in range(110):
        start = time.perf_counter()
        product(initial, direction)
        middle = time.perf_counter()
        solve(
            lambda p: execute("residual", p),
            lambda p, v: execute("vjp", p, v),
            product,
            initial,
        )
        end = time.perf_counter()
        if i >= 10:
            timings["gauss_newton_product"].append(middle - start)
            timings["complete_solve"].append(end - middle)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            name: {
                "median": float(np.median(values) * 1000),
                "p95": float(np.percentile(values, 95) * 1000),
            }
            for name, values in timings.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Isolated process; 10 warmup/100 measured repetitions; CPU FP64; product includes two sessions and host intermediate; solve includes Python CG and line search; RSS includes imports/runtime. No dense reference or export inside measurement. No acceleration comparison.",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "onnx": onnx.__version__,
        "onnxscript": importlib.metadata.version("onnxscript"),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "threads": {"intra_op": 1, "inter_op": 1},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("fitting_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(args.benchmark_worker)
            if args.benchmark_worker
            else run(args.output_dir),
            indent=2,
        )
    )
