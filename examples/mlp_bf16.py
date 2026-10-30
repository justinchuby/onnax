# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train an MLP with BF16 matrix compute and FP32 loss, gradients and master state."""

from __future__ import annotations

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
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass, ShapeInferencePass
from onnxscript import optimizer

import onnax
from examples.lora.adam import AdamState, adam_step
from examples.lora.bf16 import round_bf16
from examples.lora.bf16_rules import lower_matmuls
from examples.lora.train import close
from examples.lora.validation import discrepancy
from examples.mlp import NAMES, SHAPES, dataset, model, reference_update


def problem():
    rng = np.random.default_rng(42)
    x, y = dataset(rng, 384)
    vx, vy = dataset(rng, 192)
    params = {
        n: np.zeros(s, np.float32) if "bias" in n else rng.normal(0, 0.2, s).astype(np.float32)
        for n, s in zip(NAMES, SHAPES, strict=True)
    }
    return x, y, vx, vy, params


def reference(x, targets, params, *, precision="bf16"):
    """Independent BF16 forward and FP32 straight-through analytic backpropagation."""
    if precision not in ("bf16", "fp32"):
        raise ValueError("Expected bf16 or fp32 precision")
    quantize = round_bf16 if precision == "bf16" else np.asarray
    w1, b1, w2, b2 = (params[n] for n in NAMES)
    qx, qw1, qw2 = (quantize(v) for v in (x, w1, w2))
    z = quantize(qx @ qw1) + b1
    hidden = np.tanh(z)
    qh = quantize(hidden)
    logits = quantize(qh @ qw2) + b2
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -np.mean(np.sum(targets * logp, axis=-1))
    # Compute non-pivot entries first, then enforce logit-shift invariance.
    # This avoids subtracting one from a rounded unit probability.
    scale = np.float32(1) / np.float32(len(x))
    dz = np.exp(logp) * scale - targets * scale
    pivot = logits.argmax(axis=-1)
    dz[np.arange(len(x)), pivot] = 0
    dz[np.arange(len(x)), pivot] = -dz.sum(axis=-1)
    exponential = np.exp(np.float32(-2) * np.abs(z))
    denominator = np.float32(1) + exponential
    slope = (np.float32(4) * exponential) / (denominator * denominator)
    dh = (dz @ qw2.T) * slope
    grads = dict(
        zip(NAMES, [qx.T @ dh, dh.sum(axis=0), qh.T @ dz, dz.sum(axis=0)], strict=True)
    )
    return logits, loss, grads


def stable_log_softmax(model):
    """Expand LogSoftmax after differentiation to keep reference loss evaluation stable."""
    result = model.clone()
    for node in list(result.graph):
        if node.domain or node.op_type != "LogSoftmax":
            continue
        tape = ir.tape.Tape()
        attribute = node.attributes.get("axis")
        axis = -1 if attribute is None else attribute.as_int()
        axes = tape.op("Constant", [], {"value": ir.tensor([axis], dtype=ir.DataType.INT64)})
        maximum = tape.op("ReduceMax", [node.inputs[0], axes], {"keepdims": 1})
        shifted = tape.op("Sub", [node.inputs[0], maximum])
        summed = tape.op("ReduceSum", [tape.op("Exp", [shifted]), axes], {"keepdims": 1})
        output = tape.op("Sub", [shifted, tape.op("Log", [summed])])
        ir.convenience.replace_nodes_and_values(
            result.graph, node, [node], tape.nodes, [node.outputs[0]], [output]
        )
    return result


def graph(
    *,
    precision="bf16",
    batch=None,
    optimize=True,
    with_gradients=True,
    input_width=2,
    hidden_width=16,
    classes=3,
):
    if precision not in ("bf16", "fp32"):
        raise ValueError("Expected bf16 or fp32 precision")
    source = model(input_width=input_width, hidden_width=hidden_width, classes=classes)
    if batch is not None:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        source.graph.inputs[0].shape = ir.Shape([batch, input_width])
        source.graph.inputs[1].shape = ir.Shape([batch, classes])
        source.graph.outputs[0].shape = ir.Shape([batch, classes])
    if precision == "bf16":
        source = lower_matmuls(source)
    gradient = onnax.grad(source, inputs=NAMES, output="loss") if with_gradients else source
    inlined = InlinePass()(stable_log_softmax(gradient)).model
    inlined = ShapeInferencePass()(inlined).model
    return optimizer.optimize(inlined) if optimize else inlined


def executor(model, backend, *, profile_prefix=None):
    if backend == "reference":
        return ReferenceEvaluator(ir.to_proto(model))
    if backend == "mlx":
        from examples.lora.mlx_runtime import session

        return session(model, profile_prefix=profile_prefix)
    raise ValueError("Expected reference or mlx backend")


def run(output_dir, *, backend="reference", precision="bf16", measure=True):
    from examples.mlp import reference as fp32_reference

    os.makedirs(output_dir, exist_ok=True)
    x, y, vx, vy, initial = problem()
    raw = graph(precision=precision, batch=len(x), optimize=False)
    before = ir.to_proto(raw).SerializeToString()
    gradient = optimizer.optimize(raw.clone())
    assert ir.to_proto(raw).SerializeToString() == before
    onnx.checker.check_model(ir.to_proto(raw), full_check=True)
    ir.save(raw, os.path.join(output_dir, "gradient.raw.onnx"))
    onnx.checker.check_model(ir.to_proto(gradient), full_check=True)
    ir.save(gradient, os.path.join(output_dir, "gradient.onnx"))
    evaluate = executor(
        ir.load(os.path.join(output_dir, "gradient.onnx")),
        backend,
        profile_prefix=os.path.join(output_dir, "profile"),
    )
    validation_graph = graph(precision=precision, batch=len(vx), with_gradients=False)
    onnx.checker.check_model(ir.to_proto(validation_graph), full_check=True)
    ir.save(validation_graph, os.path.join(output_dir, "validation.onnx"))
    validation = executor(ir.load(os.path.join(output_dir, "validation.onnx")), backend)
    interface = {
        "version": 1,
        "backend": backend,
        "precision": precision,
        "loss": "loss",
        "reduction": "mean over samples; sum over classes",
        "parameters": [
            {
                "input": name,
                "gradient": value.name,
                "shape": list(initial[name].shape),
                "dtype": "float32",
            }
            for name, value in zip(NAMES, gradient.graph.outputs[2:], strict=True)
        ],
        "compute": "BF16 matrix operands/results; FP32 biases, tanh and LogSoftmax loss"
        if precision == "bf16"
        else "FP32",
        "derivatives": "FP32 local straight-through chain rule"
        if precision == "bf16"
        else "FP32 chain rule",
        "master_and_state_dtype": "float32",
        "runtime_boundary": "NumPy FP32 inputs/outputs; external CPU Array API Adam",
    }
    with open(os.path.join(output_dir, "interface.json"), "w") as stream:
        json.dump(interface, stream, indent=2)
    forward = graph(precision=precision, batch=len(x), with_gradients=False)
    onnx.checker.check_model(ir.to_proto(forward), full_check=True)
    ir.save(forward, os.path.join(output_dir, "loss.onnx"))
    params = {n: p.copy() for n, p in initial.items()}
    independent = {n: p.copy() for n, p in initial.items()}
    optimizer_reference = {n: p.copy() for n, p in initial.items()}
    optimizer_reference_state = None
    oracle = reference if precision == "bf16" else fp32_reference
    state = independent_state = None
    errors, comparisons, history = {}, {}, []
    snapshots = set()

    def compare(name, actual, expected, step):
        result = discrepancy(actual, expected)
        row = comparisons.setdefault(name, {"checks": 0, "max_abs": 0.0, "max_mismatched": 0})
        row["checks"] += 1
        row["max_abs"] = max(row["max_abs"], result["max_abs"])
        row["max_mismatched"] = max(row["max_mismatched"], result["mismatched"])
        if result["mismatched"]:
            bad = errors.setdefault(name, {"first_step": step, "steps": 0, "first": result})
            bad["steps"] += 1
            group = name.split(".")[0]
            paths = []
            if group not in snapshots:
                snapshots.add(group)
                paths.append(f"first-{group}-mismatch.npz")
            if name == "independent_trajectory.logits" and bad["steps"] == 1:
                paths.append("first-independent-logits.npz")
            for filename in paths:
                np.savez(
                    os.path.join(output_dir, filename),
                    features=x,
                    targets=y,
                    **{"native." + n: p for n, p in params.items()},
                    **{
                        "reference." + n: p
                        for n, p in (
                            optimizer_reference
                            if group == "shared_gradient_update"
                            else independent
                        ).items()
                    },
                )

    initial_validation = float(
        validation.run(None, {"features": vx, "targets": vy, **params})[1]
    )
    for step in range(150):
        actual = evaluate.run(None, {"features": x, "targets": y, **params})
        same = oracle(x, y, params)
        separate = oracle(x, y, independent)
        for group, expected in (
            ("same_parameter", same),
            ("independent_trajectory", separate),
        ):
            for name, a, b in zip(
                ("logits", "loss", *NAMES),
                actual,
                [*expected[:2], *expected[2].values()],
                strict=True,
            ):
                compare(group + "." + name, a, b, step)
        grads = dict(zip(NAMES, actual[2:], strict=True))
        params, state = adam_step(params, grads, state, xp=np)
        optimizer_reference, optimizer_reference_state = reference_update(
            optimizer_reference, grads, optimizer_reference_state
        )
        for name in NAMES:
            compare(
                "shared_gradient_update." + name,
                params[name],
                optimizer_reference[name],
                step + 1,
            )
            close(params[name], optimizer_reference[name])
        np.testing.assert_allclose(
            np.concatenate([state.first[n].reshape(-1) for n in NAMES]),
            optimizer_reference_state[1],
            rtol=1e-5,
            atol=1e-20,
        )
        np.testing.assert_allclose(
            np.concatenate([state.second[n].reshape(-1) for n in NAMES]),
            optimizer_reference_state[2],
            rtol=1e-5,
            atol=1e-20,
        )
        independent, independent_state = reference_update(
            independent, separate[2], independent_state
        )
        for name in NAMES:
            compare("parameter." + name, params[name], independent[name], step + 1)
            assert (
                params[name].dtype
                == state.first[name].dtype
                == state.second[name].dtype
                == np.float32
            )
        history.append(float(actual[1]))
    final = evaluate.run(None, {"features": x, "targets": y, **params})
    validated = validation.run(None, {"features": vx, "targets": vy, **params})
    accuracy = float(np.mean(validated[0].argmax(axis=1) == vy.argmax(axis=1)))
    assert final[1] < history[0] * 0.5 and validated[1] < initial_validation * 0.5
    assert accuracy >= 0.95
    forward_values = executor(ir.load(os.path.join(output_dir, "loss.onnx")), backend).run(
        None, {"features": x, "targets": y, **params}
    )
    for a, b in zip(final[:2], forward_values, strict=True):
        close(a, b)
    saved = {**params, "adam.step": np.array(state.step, np.int64)}
    saved.update({"adam.first." + n: state.first[n] for n in NAMES})
    saved.update({"adam.second." + n: state.second[n] for n in NAMES})
    np.savez(os.path.join(output_dir, "checkpoint.npz"), **saved)
    with np.load(os.path.join(output_dir, "checkpoint.npz")) as data:
        restored = {n: data[n] for n in NAMES}
        restored_state = AdamState(
            int(data["adam.step"]),
            {n: data["adam.first." + n] for n in NAMES},
            {n: data["adam.second." + n] for n in NAMES},
        )
    replay = executor(ir.load(os.path.join(output_dir, "gradient.onnx")), backend)
    reloaded = replay.run(None, {"features": x, "targets": y, **restored})
    for a, b in zip(final, reloaded, strict=True):
        close(a, b)
    next_params, next_state = adam_step(
        params, dict(zip(NAMES, final[2:], strict=True)), state, xp=np
    )
    resumed, resumed_state = adam_step(
        restored, dict(zip(NAMES, reloaded[2:], strict=True)), restored_state, xp=np
    )
    for n in NAMES:
        close(next_params[n], resumed[n])
        close(next_state.first[n], resumed_state.first[n])
        close(next_state.second[n], resumed_state.second[n])
    providers = []
    if backend == "mlx":
        with open(evaluate.end_profiling()) as stream:
            events = json.load(stream)
        providers = sorted(
            {
                event["args"]["provider"]
                for event in events
                if event.get("args", {}).get("provider")
            }
        )
        assert providers == ["MLXExecutionProvider"], providers
    dynamic = graph(precision=precision)
    onnx.checker.check_model(ir.to_proto(dynamic), full_check=True)
    dynamic_path = os.path.join(output_dir, "gradient.dynamic.onnx")
    ir.save(dynamic, dynamic_path)
    dynamic_replay = executor(ir.load(dynamic_path), "reference")
    for batch in (1, 7, len(vx)):
        observed = dynamic_replay.run(
            None, {"features": vx[:batch], "targets": vy[:batch], **restored}
        )
        expected = oracle(vx[:batch], vy[:batch], restored)
        for a, b in zip(observed, [*expected[:2], *expected[2].values()], strict=True):
            close(a, b)
    diagnosis = {}
    if precision == "bf16" and "independent_trajectory.logits" in errors:
        first = errors["independent_trajectory.logits"]
        with np.load(os.path.join(output_dir, "first-independent-logits.npz")) as data:
            native_point = {n: data["native." + n] for n in NAMES}
            reference_point = {n: data["reference." + n] for n in NAMES}
        index = tuple(first["first"]["first"]["index"])
        qx = round_bf16(x)

        def rounded_hidden(point):
            z = round_bf16(qx @ round_bf16(point[NAMES[0]])) + point[NAMES[1]]
            return round_bf16(np.tanh(z))

        diagnosis = {
            "step": first["first_step"],
            "index": list(index),
            "observed_logit": first["first"]["first"]["actual"],
            "same_parameter_reference_logit": float(reference(x, y, native_point)[0][index]),
            "independent_reference_logit": float(reference(x, y, reference_point)[0][index]),
            "parameter_max_abs": {
                n: float(np.max(np.abs(native_point[n] - reference_point[n]))) for n in NAMES
            },
            "rounded_weight_differences": {
                n: int(
                    np.count_nonzero(
                        round_bf16(native_point[n]) != round_bf16(reference_point[n])
                    )
                )
                for n in (NAMES[0], NAMES[2])
            },
            "rounded_hidden_differences": int(
                np.count_nonzero(
                    rounded_hidden(native_point) != rounded_hidden(reference_point)
                )
            ),
        }
    report = {
        "trajectory_diagnostic": diagnosis,
        "dynamic_reference_reload_batches": [1, 7, len(vx)],
        "executed_providers": providers,
        "backend": backend,
        "precision": precision,
        "python": platform.python_version(),
        "steps": 150,
        "strict_numerical_acceptance": not errors,
        "same_parameter_acceptance": not any(n.startswith("same_parameter.") for n in errors),
        "independent_trajectory_acceptance": not any(
            n.startswith(("independent_trajectory.", "parameter.")) for n in errors
        ),
        "shared_gradient_update_acceptance": not any(
            n.startswith("shared_gradient_update.") for n in errors
        ),
        "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "initial_train_loss": history[0],
        "final_train_loss": float(final[1]),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": float(validated[1]),
        "validation_accuracy": accuracy,
        "checkpoint_resume_verified": True,
        "parameter_bytes": sum(p.nbytes for p in params.values()),
        "state_bytes": sum(p.nbytes for p in (*state.first.values(), *state.second.values())),
        "comparisons": comparisons,
        "errors": errors,
        "loss_history": history,
        "nodes": len(gradient.graph),
        "model_sha256": hashlib.sha256(ir.to_proto(gradient).SerializeToString()).hexdigest(),
        "model_bytes": os.path.getsize(os.path.join(output_dir, "gradient.onnx")),
        "versions": {
            n: version(n)
            for n in (
                "numpy",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnxruntime",
                *(("mlx", "onnxruntime-ep-mlx") if backend == "mlx" else ()),
            )
        },
        "scope": "Offline three-class dataset; external CPU Adam; explicit FP32 host exchange for MLX; ORT optimization disabled",
    }
    np.savez(os.path.join(output_dir, "benchmark.npz"), features=x, targets=y, **initial)
    if measure:
        report["measurement"] = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "examples.mlp_bf16",
                    "--benchmark-worker",
                    str(output_dir),
                ],
                text=True,
            )
        )
    with open(os.path.join(output_dir, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


def benchmark(directory):
    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    with np.load(os.path.join(directory, "benchmark.npz")) as data:
        features, targets = data["features"], data["targets"]
        params = {name: data[name] for name in NAMES}
    started = time.perf_counter()
    model = ir.load(os.path.join(directory, "gradient.onnx"))
    loaded = time.perf_counter()
    evaluate = executor(model, interface["backend"])
    created = time.perf_counter()
    state = None
    timings = {key: [] for key in ("prepare", "execute_exchange", "adam", "step")}
    first_step_ms = None
    for index in range(110):
        started_step = time.perf_counter()
        feeds = {"features": features, "targets": targets, **params}
        prepared = time.perf_counter()
        values = evaluate.run(None, feeds)
        executed = time.perf_counter()
        params, state = adam_step(
            params, dict(zip(NAMES, values[2:], strict=True)), state, xp=np
        )
        updated = time.perf_counter()
        if index == 0:
            first_step_ms = (updated - started_step) * 1000
        if index >= 10:
            for key, value in zip(
                timings,
                (
                    prepared - started_step,
                    executed - prepared,
                    updated - executed,
                    updated - started_step,
                ),
                strict=True,
            ):
                timings[key].append(value * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "backend": interface["backend"],
        "precision": interface["precision"],
        "first_step_ms": first_step_ms,
        "model_read_ms": (loaded - started) * 1000,
        "session_creation_ms": (created - loaded) * 1000,
        "timings_ms": {
            key: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for key, v in timings.items()
        },
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "scope": "Independent worker process; same 384 samples and initial parameters; 10 warmup and 100 measured complete updates. Fetch logits/loss/all gradients. Execution includes synchronous host exchange, MLX internal copies and result allocation. Adam runs on CPU. RSS includes imports, loaded model, runtime and allocator overhead.",
        "threads": {"ort_intra_op": 1, "ort_inter_op": 1}
        if interface["backend"] == "mlx"
        else None,
        "array_thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="mlp_bf16_models")
    parser.add_argument("--benchmark-worker", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["reference", "mlx"], default="reference")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    if args.benchmark_worker:
        print(json.dumps(benchmark(args.benchmark_worker)))
        raise SystemExit(0)
    report = run(args.output_dir, backend=args.backend, precision=args.precision)
    print(
        f"Loss: {report['initial_train_loss']:.6g} -> {report['final_train_loss']:.6g}; validation accuracy: {report['validation_accuracy']:.3f}"
    )
    print(f"Strict numerical acceptance: {report['strict_numerical_acceptance']}")
    raise SystemExit(0 if report["strict_numerical_acceptance"] else 1)
