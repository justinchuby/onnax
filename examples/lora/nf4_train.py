# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train LoRA adapters over a packed NF4 base using portable ONNX reference kernels."""

import argparse
import hashlib
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

import onnax
from examples.lora.adam import AdamState, adam_step
from examples.lora.model import PARAMETERS, adapt, base_model, with_loss
from examples.lora.nf4 import CODEBOOK, dequantize, lower
from examples.lora.train import (
    finite_differences,
    problem,
    reference,
    reference_update,
    session,
)


def frozen_digest(model, manifest):
    return {
        name: hashlib.sha256(
            model.graph.initializers[name].const_value.numpy().tobytes()
        ).hexdigest()
        for layer in manifest.values()
        for name in (layer["packed"], layer["scales"])
    }


def execute(executor, x, y, params, mapping):
    values = executor.run(None, {"features": x, "target": y, **params})
    outputs = dict(zip((v.name for v in executor.get_outputs()), values, strict=True))
    return values[0], values[1], {name: outputs[output] for name, output in mapping.items()}


def benchmark(directory):
    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    with np.load(os.path.join(directory, "benchmark.npz")) as data:
        x, y = data["features"], data["target"]
        params = {n: data[n].copy() for n in PARAMETERS}
    started = time.perf_counter()
    executor = session(os.path.join(directory, "gradient.optimized.onnx"))
    creation = (time.perf_counter() - started) * 1000
    samples = []
    state = None
    for index in range(110):
        started = time.perf_counter()
        _, _, grads = execute(executor, x, y, params, interface["gradients"])
        executed = time.perf_counter()
        params, state = adam_step(params, grads, state, xp=np)
        finished = time.perf_counter()
        if index >= 10:
            samples.append(
                {
                    "gradient": (executed - started) * 1000,
                    "adam": (finished - executed) * 1000,
                    "step": (finished - started) * 1000,
                }
            )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "session_creation_ms": creation,
        "warmup": 10,
        "samples": 100,
        "timings_ms": {
            n: {
                "median": float(np.median([s[n] for s in samples])),
                "p95": float(np.percentile([s[n] for s in samples], 95)),
            }
            for n in samples[0]
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "scope": "Separate process; CPU ONNX NF4 decode/MatMul reference kernels, NumPy Adam; includes process/runtime overhead",
        "threads": {"intra_op": 1, "inter_op": 1},
        "ort_optimizer": "disabled",
    }


def run(directory, *, measure=True):
    os.makedirs(directory, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    base_path = os.path.join(directory, "base.fp32.onnx")
    ir.save(base_model(w, c), base_path)
    original = adapt(ir.load(base_path))
    forward, manifest = lower(original)
    expected_frozen = frozen_digest(forward, manifest)
    weights = {
        name: dequantize(
            forward.graph.initializers[item["packed"]].const_value.numpy(),
            forward.graph.initializers[item["scales"]].const_value.numpy(),
            item["shape"],
            block_size=item["block_size"],
        )
        for name, item in manifest.items()
    }
    loss = with_loss(forward)
    gradient = onnax.grad(loss, inputs=list(PARAMETERS), output="loss")
    optimized = optimizer.optimize(gradient.clone(), inline=False)
    mapping = dict(zip(PARAMETERS, [v.name for v in gradient.graph.outputs[2:]], strict=True))
    for name, model in [
        ("forward", forward),
        ("gradient", gradient),
        ("gradient.optimized", optimized),
    ]:
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        assert frozen_digest(model, manifest) == expected_frozen
        assert not {"base.W", "base.C"}.intersection(model.graph.initializers)
        ir.save(model, os.path.join(directory, name + ".onnx"))
    interface = {
        "version": 1,
        "quantization": "NF4",
        "packing": "row-major, high nibble first, zero-padded blocks",
        "codebook": CODEBOOK.tolist(),
        "compute_dtype": "float32",
        "gradient_dtype": "float32",
        "master_and_state_dtype": "float32",
        "parameters": list(PARAMETERS),
        "gradients": mapping,
        "base": manifest,
        "frozen_sha256": expected_frozen,
        "loss": "loss",
        "reduction": "mean over samples and output features",
        "runtime": "CPU ONNX reference NF4 kernels",
    }
    with open(os.path.join(directory, "interface.json"), "w") as stream:
        json.dump(interface, stream, indent=2)
    np.savez(os.path.join(directory, "benchmark.npz"), features=x, target=y, **params)
    executor = session(os.path.join(directory, "gradient.optimized.onnx"))
    raw = session(os.path.join(directory, "gradient.onnx"))
    oracle = {n: p.copy() for n, p in params.items()}
    state = oracle_state = None
    history = []
    initial_validation = float(
        reference(vx, vy, params, weights["base.W"], weights["base.C"])[1]
    )
    for index in range(200):
        prediction, value, grads = execute(executor, x, y, params, mapping)
        expected = reference(x, y, params, weights["base.W"], weights["base.C"])
        np.testing.assert_allclose(prediction, expected[0], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(value, expected[1], rtol=1e-4, atol=1e-6)
        for n in PARAMETERS:
            np.testing.assert_allclose(grads[n], expected[2][n], rtol=1e-4, atol=1e-6)
        if index == 0:
            assert (
                not np.any(grads[PARAMETERS[0]]) and np.linalg.norm(grads[PARAMETERS[1]]) > 0
            )
        if index == 1:
            assert np.linalg.norm(grads[PARAMETERS[0]]) > 0
            finite_differences(x, y, params, weights["base.W"], weights["base.C"])
        _, oracle_loss, oracle_grads = reference(
            x, y, oracle, weights["base.W"], weights["base.C"]
        )
        np.testing.assert_allclose(value, oracle_loss, rtol=1e-4, atol=1e-6)
        oracle, oracle_state = reference_update(oracle, oracle_grads, oracle_state)
        params, state = adam_step(params, grads, state, xp=np)
        for n in PARAMETERS:
            np.testing.assert_allclose(params[n], oracle[n], rtol=1e-4, atol=1e-6)
        history.append(float(value))
    assert state.step == 200
    final_loss = float(reference(x, y, params, weights["base.W"], weights["base.C"])[1])
    final_validation = float(
        reference(vx, vy, params, weights["base.W"], weights["base.C"])[1]
    )
    assert final_loss < history[0] * 0.5 and final_validation < initial_validation * 0.5
    saved = {
        **params,
        "adam.step": np.array(state.step, np.int64),
        **{f"adam.first.{n}": state.first[n] for n in PARAMETERS},
        **{f"adam.second.{n}": state.second[n] for n in PARAMETERS},
    }
    np.savez(os.path.join(directory, "checkpoint.npz"), **saved)
    with np.load(os.path.join(directory, "checkpoint.npz")) as data:
        restored = {n: data[n].copy() for n in PARAMETERS}
        restored_state = AdamState(
            int(data["adam.step"]),
            {n: data[f"adam.first.{n}"].copy() for n in PARAMETERS},
            {n: data[f"adam.second.{n}"].copy() for n in PARAMETERS},
        )
    fresh = session(os.path.join(directory, "gradient.optimized.onnx"))
    for rows in (1, 7, 128):
        a = execute(fresh, vx[:rows], vy[:rows], restored, mapping)
        b = execute(raw, vx[:rows], vy[:rows], params, mapping)
        for av, bv in [(a[0], b[0]), (a[1], b[1]), *[(a[2][n], b[2][n]) for n in PARAMETERS]]:
            np.testing.assert_allclose(av, bv, rtol=1e-4, atol=1e-6)
    a = execute(executor, x, y, params, mapping)
    b = execute(fresh, x, y, restored, mapping)
    updated, next_state = adam_step(params, a[2], state, xp=np)
    resumed, resumed_state = adam_step(restored, b[2], restored_state, xp=np)
    for n in PARAMETERS:
        for av, bv in [
            (updated[n], resumed[n]),
            (next_state.first[n], resumed_state.first[n]),
            (next_state.second[n], resumed_state.second[n]),
        ]:
            assert av.tobytes() == bv.tobytes()
    assert next_state.step == resumed_state.step == 201
    for name in ("forward", "gradient", "gradient.optimized"):
        assert (
            frozen_digest(ir.load(os.path.join(directory, name + ".onnx")), manifest)
            == expected_frozen
        )
    restored_prediction = session(os.path.join(directory, "forward.onnx")).run(
        None, {"features": vx, **restored}
    )[0]
    np.testing.assert_allclose(
        restored_prediction,
        reference(vx, vy, params, weights["base.W"], weights["base.C"])[0],
        rtol=1e-4,
        atol=1e-6,
    )
    report = {
        "version": 1,
        "steps": 200,
        "train_loss": {"initial": history[0], "final": final_loss},
        "validation_loss": {"initial": initial_validation, "final": final_validation},
        "history": history,
        "frozen_weights_unchanged": True,
        "adapter_checkpoint_replay": True,
        "dynamic_batches": [1, 7, 128],
        "base_original_bytes": sum(v["original_bytes"] for v in manifest.values()),
        "base_packed_bytes": sum(
            v["packed_bytes"] + v["scale_bytes"] for v in manifest.values()
        ),
        "storage_scope": "Frozen matrix codes and FP32 block scales; excludes codebooks, graph metadata and temporary dequantized buffers",
        "execution": "Portable CPU NF4 decode and dense MatMul; no fused GPU NF4 kernel",
        "platform": platform.platform(),
        "versions": {
            n: version(n) for n in ("numpy", "onnx", "onnx-ir", "onnxruntime", "onnxscript")
        },
    }
    if measure:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.lora.nf4_train",
                "--benchmark-only",
                "--output-dir",
                directory,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["measurement"] = json.loads(result.stdout)
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="nf4_lora_models")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps((benchmark if args.benchmark_only else run)(args.output_dir), indent=2))
