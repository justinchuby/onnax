# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train an ONNX MLP classifier: python -m examples.mlp --output-dir mlp_models."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import array_api_compat
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import optimizer

import onnax
from examples.lora.adam import adam_step
from examples.lora.train import close, session

NAMES = ("layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias")
SHAPES = ([2, 16], [16], [16, 3], [3])


def model(*, input_width=2, hidden_width=16, classes=3):
    for name, value in (
        ("input_width", input_width),
        ("hidden_width", hidden_width),
        ("classes", classes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    shapes = ([input_width, hidden_width], [hidden_width], [hidden_width, classes], [classes])
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["batch", input_width])
    target = ir.val("targets", dtype=ir.DataType.FLOAT, shape=["batch", classes])
    params = [
        ir.val(n, dtype=ir.DataType.FLOAT, shape=s) for n, s in zip(NAMES, shapes, strict=True)
    ]
    tape = ir.tape.Tape()
    hidden = tape.op("Tanh", [tape.op("Add", [tape.op("MatMul", [x, params[0]]), params[1]])])
    logits = tape.op("Add", [tape.op("MatMul", [hidden, params[2]]), params[3]])
    logits.name = "logits"
    logits.type = ir.TensorType(ir.DataType.FLOAT)
    logits.shape = ir.Shape(["batch", classes])
    logp = tape.op("LogSoftmax", [logits], {"axis": -1})
    axes = tape.initializer(ir.tensor(np.array([-1], np.int64), name="class_axis"))
    per_sample = tape.op(
        "Neg", [tape.op("ReduceSum", [tape.op("Mul", [target, logp]), axes], {"keepdims": 0})]
    )
    loss = tape.op("ReduceMean", [per_sample], {"keepdims": 0})
    loss.name = "loss"
    loss.type = ir.TensorType(ir.DataType.FLOAT)
    loss.shape = ir.Shape([])
    return ir.Model(
        ir.Graph(
            [x, target, *params],
            [logits, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="mlp_classifier",
        ),
        ir_version=10,
    )


def reference(x, targets, p):
    w1, b1, w2, b2 = (p[n] for n in NAMES)
    h = np.tanh(x @ w1 + b1)
    logits = h @ w2 + b2
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logp = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    loss = -np.mean(np.sum(targets * logp, axis=-1))
    # Compute non-pivot entries first, then enforce logit-shift invariance.
    # This avoids subtracting one from a rounded unit probability.
    scale = np.float32(1) / np.float32(len(x))
    dz = np.exp(logp) * scale - targets * scale
    pivot = logits.argmax(axis=-1)
    dz[np.arange(len(x)), pivot] = 0
    dz[np.arange(len(x)), pivot] = -dz.sum(axis=-1)
    dh = (dz @ w2.T) * (1 - h * h)
    grads = dict(zip(NAMES, [x.T @ dh, dh.sum(axis=0), h.T @ dz, dz.sum(axis=0)], strict=True))
    return logits, loss, grads


def reference_update(params, grads, state, *, lr=0.03, eps=1e-8, beta1=0.9, beta2=0.999):
    vector = np.concatenate([params[n].reshape(-1) for n in NAMES])
    gradient = np.concatenate([grads[n].reshape(-1) for n in NAMES])
    step, mean, variance = (
        (0, np.zeros_like(vector), np.zeros_like(vector)) if state is None else state
    )
    step += 1
    mean = beta1 * mean + (1 - beta1) * gradient
    variance = beta2 * variance + (1 - beta2) * gradient**2
    vector -= lr * (mean / (1 - beta1**step)) / (np.sqrt(variance / (1 - beta2**step)) + eps)
    chunks = np.split(vector, np.cumsum([params[n].size for n in NAMES])[:-1])
    return {
        n: value.reshape(params[n].shape) for n, value in zip(NAMES, chunks, strict=True)
    }, (step, mean, variance)


def dataset(rng, count):
    labels = rng.integers(0, 3, size=count)
    centers = np.array([[-2, -1], [2, -1], [0, 2]], np.float32)
    x = centers[labels] + rng.normal(0, 0.55, (count, 2)).astype(np.float32)
    return x, np.eye(3, dtype=np.float32)[labels]


def benchmark(directory):
    directory = Path(directory)
    data = np.load(directory / "benchmark.npz")
    interface = json.loads((directory / "interface.json").read_text())
    params = {n: data[n] for n in NAMES}
    started = time.perf_counter()
    executor = session(directory / "gradient.optimized.onnx")
    creation = time.perf_counter() - started
    xp = array_api_compat.array_namespace(*params.values())
    state = None
    names = [row["gradient"] for row in interface["parameters"]]
    times = {key: [] for key in ("gradient", "adam", "step")}
    for i in range(110):
        start = time.perf_counter()
        gradients = executor.run(
            names, {"features": data["features"], "targets": data["targets"], **params}
        )
        evaluated = time.perf_counter()
        params, state = adam_step(
            params, dict(zip(NAMES, gradients, strict=True)), state, xp=xp
        )
        updated = time.perf_counter()
        if i >= 10:
            times["gradient"].append(evaluated - start)
            times["adam"].append(updated - evaluated)
            times["step"].append(updated - start)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            k: {
                "median": float(np.median(v) * 1000),
                "p95": float(np.percentile(v, 95) * 1000),
            }
            for k, v in times.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process; CPU FP32; gradients-only outputs; 10 warmup/100 measured updates. RSS includes imports/runtime, not tensor peak. Gradient time includes CPU feed preparation and output arrays. No GPU transfer or acceleration comparison.",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "threads": {"intra_op": 1, "inter_op": 1},
    }


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x, y = dataset(rng, 384)
    vx, vy = dataset(rng, 192)
    params = {
        n: (
            np.zeros(s, np.float32)
            if "bias" in n
            else rng.normal(0, 0.2, s).astype(np.float32)
        )
        for n, s in zip(NAMES, SHAPES, strict=True)
    }
    independent = {n: v.copy() for n, v in params.items()}
    source = model()
    before = ir.to_proto(source).SerializeToString()
    gradient = onnax.grad(source, inputs=list(NAMES), output="loss")
    mapping = dict(zip(NAMES, [v.name for v in gradient.graph.outputs[2:]], strict=True))
    for label, graph in (("loss", source), ("gradient", gradient)):
        proto = ir.to_proto(graph)
        onnx.checker.check_model(proto, full_check=True)
        onnx.save(proto, output_dir / f"{label}.onnx")
    optimized = optimizer.optimize(ir.to_proto(gradient))
    onnx.checker.check_model(optimized, full_check=True)
    onnx.save(optimized, output_dir / "gradient.optimized.onnx")
    interface = {
        "version": 1,
        "loss": "loss",
        "reduction": "mean over samples; sum over classes",
        "parameters": [
            {
                "input": n,
                "gradient": mapping[n],
                "shape": list(params[n].shape),
                "dtype": "float32",
            }
            for n in NAMES
        ],
    }
    (output_dir / "interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    execute = session(output_dir / "gradient.optimized.onnx")
    xp = array_api_compat.array_namespace(*params.values())
    state = ref_state = None
    initial = float(reference(x, y, params)[1])
    initial_validation = float(reference(vx, vy, params)[1])
    history = []
    for _step in range(150):
        result = execute.run(
            ["logits", "loss", *mapping.values()], {"features": x, "targets": y, **params}
        )
        # Isolate graph semantics at identical parameters from trajectory drift.
        at_same_parameters = reference(x, y, params)
        close(result[0], at_same_parameters[0])
        close(result[1], at_same_parameters[1])
        for n, value in zip(NAMES, result[2:], strict=True):
            close(value, at_same_parameters[2][n])
        expected = reference(x, y, independent)
        close(result[1], expected[1])
        grads = dict(zip(NAMES, result[2:], strict=True))
        params, state = adam_step(params, grads, state, xp=xp)
        independent, ref_state = reference_update(independent, expected[2], ref_state)
        for n in NAMES:
            close(params[n], independent[n])
        history.append(float(result[1]))
    raw = session(output_dir / "gradient.onnx")
    loaded = session(output_dir / "gradient.optimized.onnx")
    for batch in (1, 7, 192):
        expected = reference(vx[:batch], vy[:batch], params)
        for executor in (raw, loaded):
            values = executor.run(
                ["logits", "loss", *mapping.values()],
                {"features": vx[:batch], "targets": vy[:batch], **params},
            )
            close(values[0], expected[0])
            close(values[1], expected[1])
            for n, value in zip(NAMES, values[2:], strict=True):
                close(value, expected[2][n])
    prediction, validation, _ = reference(vx, vy, params)
    final = float(reference(x, y, params)[1])
    accuracy = float(np.mean(prediction.argmax(axis=1) == vy.argmax(axis=1)))
    assert final < initial * 0.5 and validation < initial_validation * 0.5
    assert accuracy >= 0.95
    assert before == ir.to_proto(source).SerializeToString()
    np.savez(output_dir / "parameters.npz", **params)
    restored = dict(np.load(output_dir / "parameters.npz"))
    restored_prediction = session(output_dir / "loss.onnx").run(
        ["logits"], {"features": vx, "targets": vy, **restored}
    )[0]
    close(restored_prediction, prediction)
    report = {
        "steps": 150,
        "initial_train_loss": initial,
        "final_train_loss": final,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": float(validation),
        "validation_accuracy": accuracy,
        "loss_history": history,
        "scope": "Offline three-class synthetic dataset; CPUExecutionProvider; FP32; ORT optimization disabled. No performance comparison.",
    }
    report["parameter_bytes"] = sum(value.nbytes for value in params.values())
    report["artifacts"] = {
        name: {
            "bytes": (output_dir / f"{name}.onnx").stat().st_size,
            "nodes": len(ir.load(output_dir / f"{name}.onnx").graph),
        }
        for name in ("loss", "gradient", "gradient.optimized")
    }
    np.savez(output_dir / "benchmark.npz", features=x, targets=y, **params)
    report["measurement"] = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "examples.mlp", "--benchmark-worker", str(output_dir)],
            text=True,
        )
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("mlp_models"))
    parser.add_argument("--benchmark-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.benchmark_worker:
        print(json.dumps(benchmark(args.benchmark_worker)))
    else:
        result = run(args.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))
