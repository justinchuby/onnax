# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Scale classifier training and compare CPU Adam with shared-buffer Metal updates."""

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
from functools import partial
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

from examples.lora.adam import AdamState, adam_step
from examples.lora.mlx_adam import CompiledAdam
from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import host
from examples.lora.train import close
from examples.lora.validation import discrepancy
from examples.mlp import NAMES, reference_update
from examples.mlp_bf16 import graph, reference
from examples.mlp_bf16 import problem as tiny_problem
from examples.mlx_binding import MlxBinding, synchronize

# input width, hidden width, classes, training rows, validation rows
PROFILES = {
    "tiny": (2, 16, 3, 384, 192),
    "medium": (128, 512, 16, 1024, 256),
    "large": (512, 2048, 32, 2048, 512),
    "activation": (128, 2048, 16, 16384, 512),
}


def problem(profile):
    if profile not in PROFILES:
        raise ValueError(f"Expected a profile in {sorted(PROFILES)}")
    if profile == "tiny":
        return tiny_problem()
    width, hidden, classes, rows, validation_rows = PROFILES[profile]
    rng = np.random.default_rng(20261030)
    centers = rng.normal(size=(classes, width)).astype(np.float32)
    labels = rng.integers(0, classes, rows + validation_rows)
    x = centers[labels] + rng.normal(0, 0.75, (len(labels), width)).astype(np.float32)
    targets = np.float32(0.875) * np.eye(classes, dtype=np.float32)[labels] + np.float32(
        0.125 / classes
    )
    params = dict(
        zip(
            NAMES,
            (
                rng.normal(0, 0.2 / np.sqrt(width), (width, hidden)).astype(np.float32),
                np.zeros(hidden, np.float32),
                rng.normal(0, 0.2 / np.sqrt(hidden), (hidden, classes)).astype(np.float32),
                np.zeros(classes, np.float32),
            ),
            strict=True,
        )
    )
    return x[:rows], targets[:rows], x[rows:], targets[rows:], params


def models(profile, precision):
    if profile not in PROFILES:
        raise ValueError(f"Expected a profile in {sorted(PROFILES)}")
    width, hidden, classes, batch, validation_batch = PROFILES[profile]
    options = dict(
        precision=precision, input_width=width, hidden_width=hidden, classes=classes
    )
    raw = graph(**options, batch=batch, optimize=False)
    raw.graph.outputs.remove(raw.graph.outputs[0])
    train = optimizer.optimize(raw)
    forward = graph(**options, batch=batch, with_gradients=False)
    validation = graph(**options, batch=validation_batch, with_gradients=False)
    return train, forward, validation


def execute(executor, binding, data, params):
    started = time.perf_counter()
    feeds = {**data, **params}
    if binding is not None:
        binding.bind(feeds)
    prepared = time.perf_counter()
    if binding is not None:
        binding.execute()
    else:
        values = executor.run(None, feeds)
    executed = time.perf_counter()
    outputs = (
        binding.import_outputs()
        if binding is not None
        else dict(zip([v.name for v in executor.get_outputs()], values, strict=True))
    )
    imported = time.perf_counter()
    return outputs, (started, prepared, executed, imported)


def step(executor, binding, data, params, state, update, mapping):
    outputs, times = execute(executor, binding, data, params)
    grads = {name: outputs[output] for name, output in mapping.items()}
    updated, state = update(params, grads, state)
    if binding is not None:
        synchronize(updated, state)
    finished = time.perf_counter()
    started, prepared, executed, imported = times
    return (
        outputs["loss"],
        grads,
        updated,
        state,
        {
            "prepare": (prepared - started) * 1000,
            "execute_exchange": (executed - prepared) * 1000,
            "import": (imported - executed) * 1000,
            "adam_sync": (finished - imported) * 1000,
            "step": (finished - started) * 1000,
        },
    )


def setup(interface, initial, x, y):
    import mlx.core as mx

    mode = interface["mode"]
    if mode not in ("cpu", "eager", "compiled"):
        raise ValueError("Expected cpu, eager or compiled mode")
    if mode == "cpu":
        return (
            {"features": x, "targets": y},
            {n: p.copy() for n, p in initial.items()},
            partial(adam_step, xp=np, **interface["adam"]),
        )
    data = {"features": mx.array(x), "targets": mx.array(y)}
    params = {n: mx.array(p) for n, p in initial.items()}
    synchronize({**data, **params})
    update = (
        CompiledAdam(**interface["adam"])
        if mode == "compiled"
        else partial(adam_step, xp=mx, **interface["adam"])
    )
    return data, params, update


def cpu_values(values, mode):
    return values if mode == "cpu" else {n: host(v) for n, v in values.items()}


def run(directory, *, profile="medium", precision="fp32", mode="compiled", measure=True):
    import mlx.core as mx

    os.makedirs(directory, exist_ok=True)
    x, y, vx, vy, initial = problem(profile)
    train, forward, validation = models(profile, precision)
    for name, model in (("gradient", train), ("forward", forward), ("validation", validation)):
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        ir.save(model, os.path.join(directory, name + ".onnx"))
    mapping = dict(zip(NAMES, [v.name for v in train.graph.outputs[1:]], strict=True))
    interface = {
        "version": 1,
        "profile": profile,
        "precision": precision,
        "mode": mode,
        "dimensions": dict(
            zip(("input", "hidden", "classes"), PROFILES[profile][:3], strict=True)
        ),
        "data_shapes": {"features": list(x.shape), "targets": list(y.shape)},
        "validation_rows": len(vx),
        "adam": {
            "lr": 0.03 if profile == "tiny" else 0.01,
            "eps": 1e-8 if profile == "tiny" else 1e-6,
            "beta1": 0.9,
            "beta2": 0.999,
        },
        "parameters": [
            {"input": n, "gradient": mapping[n], "shape": list(p.shape), "dtype": "float32"}
            for n, p in initial.items()
        ],
        "label_smoothing": 0.0 if profile == "tiny" else 0.125,
        "loss": "loss",
        "reduction": "mean over samples; sum over classes",
    }
    with open(os.path.join(directory, "interface.json"), "w") as stream:
        json.dump(interface, stream, indent=2)
    executor = session(train, profile_prefix=os.path.join(directory, "profile"))
    forward_executor, validation_executor = session(forward), session(validation)
    shared = mode != "cpu"
    bound = MlxBinding(executor) if shared else None
    forward_bound = MlxBinding(forward_executor) if shared else None
    validation_bound = MlxBinding(validation_executor) if shared else None
    errors, comparisons, history = {}, {}, []
    snapshots = {}
    independent = {n: p.copy() for n, p in initial.items()}
    shared_reference = {n: p.copy() for n, p in initial.items()}
    state = independent_state = shared_state = None

    def compare(name, actual, expected, index):
        row = discrepancy(actual, expected)
        summary = comparisons.setdefault(
            name, {"checks": 0, "max_abs": 0.0, "max_mismatched": 0}
        )
        summary["checks"] += 1
        summary["max_abs"] = max(summary["max_abs"], row["max_abs"])
        summary["max_mismatched"] = max(summary["max_mismatched"], row["mismatched"])
        if row["mismatched"]:
            bad = errors.setdefault(name, {"first_step": index, "steps": 0, "first": row})
            bad["steps"] += 1
            group = name.split(".")[0]
            if group not in snapshots:
                filename = f"first-{group}-mismatch.npz"
                snapshots[group] = {"comparison": name, "step": index, "file": filename}
                reference_point = (
                    point
                    if group == "same_parameter"
                    else shared_reference
                    if group == "shared_gradient_update"
                    else independent
                )
                np.savez(
                    os.path.join(directory, filename),
                    features=x,
                    targets=y,
                    **{"native." + n: p for n, p in point.items()},
                    **{"reference." + n: p for n, p in reference_point.items()},
                )

    with mx.stream(mx.gpu):
        data, params, update = setup(interface, initial, x, y)
        validation_data = (
            {"features": mx.array(vx), "targets": mx.array(vy)}
            if shared
            else {"features": vx, "targets": vy}
        )
        initial_validation = float(
            cpu_values(
                execute(validation_executor, validation_bound, validation_data, params)[0],
                mode,
            )["loss"]
        )
        for index in range(150):
            point = {n: p.copy() for n, p in cpu_values(params, mode).items()}
            loss, grads, next_params, next_state, _ = step(
                executor, bound, data, params, state, update, mapping
            )
            for n, p in cpu_values(params, mode).items():
                np.testing.assert_array_equal(p, point[n])
            observed = cpu_values({"loss": loss, **grads}, mode)
            same = reference(x, y, point, precision=precision)
            separate = reference(x, y, independent, precision=precision)
            predicted = cpu_values(
                execute(forward_executor, forward_bound, data, params)[0], mode
            )
            for prefix, expected in (
                ("same_parameter", same),
                ("independent_trajectory", separate),
            ):
                for name, a, b in zip(
                    ("logits", "loss", *NAMES),
                    [predicted["logits"], observed["loss"], *[observed[n] for n in NAMES]],
                    [*expected[:2], *expected[2].values()],
                    strict=True,
                ):
                    compare(prefix + "." + name, a, b, index)
            shared_reference, shared_state = reference_update(
                shared_reference,
                {n: observed[n] for n in NAMES},
                shared_state,
                **interface["adam"],
            )
            independent, independent_state = reference_update(
                independent, separate[2], independent_state, **interface["adam"]
            )
            params, state = next_params, next_state
            point = cpu_values(params, mode)
            for n in NAMES:
                compare(
                    "shared_gradient_update." + n, point[n], shared_reference[n], index + 1
                )
                compare("parameter." + n, point[n], independent[n], index + 1)
                if shared:
                    assert (
                        params[n].__dlpack_device__()
                        == state.first[n].__dlpack_device__()
                        == state.second[n].__dlpack_device__()
                        == (8, 0)
                    )
                assert (
                    params[n].dtype
                    == state.first[n].dtype
                    == state.second[n].dtype
                    == (mx.float32 if shared else np.float32)
                )
            for actual, expected in (
                (state.first, shared_state[1]),
                (state.second, shared_state[2]),
            ):
                moments = cpu_values(actual, mode)
                flat = np.concatenate([moments[n].reshape(-1) for n in NAMES])
                np.testing.assert_allclose(flat, expected, rtol=1e-5, atol=1e-15)
            history.append(float(observed["loss"]))
        final = cpu_values(execute(forward_executor, forward_bound, data, params)[0], mode)
        validated = cpu_values(
            execute(validation_executor, validation_bound, validation_data, params)[0], mode
        )
        accuracy = float(np.mean(validated["logits"].argmax(axis=-1) == vy.argmax(axis=-1)))
        assert (
            final["loss"] < history[0] * 0.5 and validated["loss"] < initial_validation * 0.5
        )
        assert accuracy >= 0.95
        checkpoint = {**cpu_values(params, mode), "adam.step": np.array(state.step, np.int64)}
        checkpoint.update(
            {"adam.first." + n: p for n, p in cpu_values(state.first, mode).items()}
        )
        checkpoint.update(
            {"adam.second." + n: p for n, p in cpu_values(state.second, mode).items()}
        )
        np.savez(os.path.join(directory, "checkpoint.npz"), **checkpoint)
        with np.load(os.path.join(directory, "checkpoint.npz")) as saved:
            convert = mx.array if shared else np.array
            restored = {n: convert(saved[n]) for n in NAMES}
            restored_state = AdamState(
                int(saved["adam.step"]),
                {n: convert(saved["adam.first." + n]) for n in NAMES},
                {n: convert(saved["adam.second." + n]) for n in NAMES},
            )
        fresh = session(ir.load(os.path.join(directory, "gradient.onnx")))
        fresh_bound = MlxBinding(fresh) if shared else None
        before = execute(executor, bound, data, params)[0]
        replay = execute(fresh, fresh_bound, data, restored)[0]
        replay_host = cpu_values(replay, mode)
        for n, a in cpu_values(before, mode).items():
            close(a, replay_host[n])
        restored_update = CompiledAdam(**interface["adam"]) if mode == "compiled" else update
        updated, updated_state = update(
            params, {n: before[g] for n, g in mapping.items()}, state
        )
        replayed, replayed_state = restored_update(
            restored, {n: replay[g] for n, g in mapping.items()}, restored_state
        )
        if shared:
            synchronize(updated, updated_state)
            synchronize(replayed, replayed_state)
        for a, b in (
            (updated, replayed),
            (updated_state.first, replayed_state.first),
            (updated_state.second, replayed_state.second),
        ):
            right = cpu_values(b, mode)
            for n, p in cpu_values(a, mode).items():
                close(p, right[n])
    with open(executor.end_profiling()) as stream:
        providers = sorted(
            {
                e["args"]["provider"]
                for e in json.load(stream)
                if e.get("args", {}).get("provider")
            }
        )
    assert providers == ["MLXExecutionProvider"]
    report = {
        **interface,
        "steps": 150,
        "strict_numerical_acceptance": not errors,
        "same_parameter_acceptance": not any(n.startswith("same_parameter.") for n in errors),
        "shared_gradient_update_acceptance": not any(
            n.startswith("shared_gradient_update.") for n in errors
        ),
        "independent_trajectory_acceptance": not any(
            n.startswith(("independent_trajectory.", "parameter.")) for n in errors
        ),
        "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "snapshots": snapshots,
        "comparisons": comparisons,
        "errors": errors,
        "initial_train_loss": history[0],
        "final_train_loss": float(final["loss"]),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": float(validated["loss"]),
        "validation_accuracy": accuracy,
        "loss_history": history,
        "checkpoint_resume_verified": True,
        "master_and_state_on_metal": shared,
        "pointer_identity_checked": shared,
        "parameter_bytes": sum(p.nbytes for p in initial.values()),
        "state_bytes": 2 * sum(p.nbytes for p in initial.values()),
        "python_boundary_copy_bytes_per_step": 0 if shared else None,
        "model_bytes": os.path.getsize(os.path.join(directory, "gradient.onnx")),
        "nodes": len(train.graph),
        "model_sha256": hashlib.sha256(ir.to_proto(train).SerializeToString()).hexdigest(),
        "executed_providers": providers,
    }
    np.savez(os.path.join(directory, "benchmark.npz"), features=x, targets=y, **initial)
    if measure:
        report["measurement"] = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "examples.mlp_device",
                    "--benchmark-worker",
                    str(directory),
                ],
                text=True,
            )
        )
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


def benchmark(directory):
    import mlx.core as mx

    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    with np.load(os.path.join(directory, "benchmark.npz")) as data:
        x, y = data["features"], data["targets"]
        initial = {n: data[n] for n in NAMES}
    model = ir.load(os.path.join(directory, "gradient.onnx"))
    started = time.perf_counter()
    executor = session(model)
    creation = time.perf_counter() - started
    mapping = {row["input"]: row["gradient"] for row in interface["parameters"]}
    binding = MlxBinding(executor) if interface["mode"] != "cpu" else None
    times = {
        name: [] for name in ("prepare", "execute_exchange", "import", "adam_sync", "step")
    }
    with mx.stream(mx.gpu):
        data, params, update = setup(interface, initial, x, y)
        state = None
        active = mx.get_active_memory()
        mx.reset_peak_memory()
        first_step = None
        for index in range(110):
            _, _, params, state, metrics = step(
                executor, binding, data, params, state, update, mapping
            )
            if index == 0:
                first_step = metrics["step"]
            if index >= 10:
                for name, value in metrics.items():
                    times[name].append(value)
        memory = {
            "active": mx.get_active_memory(),
            "peak": max(active, mx.get_peak_memory()),
            "cache": mx.get_cache_memory(),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "session_creation_ms": creation * 1000,
        "first_step_ms": first_step,
        "timings_ms": {
            n: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for n, v in times.items()
        },
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "python_mlx_allocator_bytes": memory,
        "returned_output_bytes_per_step": 4 + sum(p.nbytes for p in initial.values()),
        "scope": "Independent worker; ten warmup/100 measured full updates. Prepare includes CPU feeds or synchronized Metal binding; execute includes provider wrapping/copy-out; import checks shared output pointers; Adam includes synchronization of parameters and moments. References/checkpoints excluded. RSS covers process/runtime. MLX counters cover the Python allocator and its active baseline; provider allocations use a separate runtime.",
        "threads": {"intra_op": 1, "inter_op": 1},
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in (
                "numpy",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnxruntime",
                "mlx",
                "onnxruntime-ep-mlx",
            )
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="mlp_device_models")
    parser.add_argument("--profile", choices=list(PROFILES), default="medium")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--mode",
        choices=["cpu", "eager", "compiled"],
        default="compiled",
        help="CPU Adam or eager/compiled Metal Adam; all model graphs execute on MLX",
    )
    parser.add_argument("--benchmark-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.benchmark_worker:
        print(json.dumps(benchmark(args.benchmark_worker)))
        raise SystemExit(0)
    report = run(
        args.output_dir, profile=args.profile, precision=args.precision, mode=args.mode
    )
    print(
        f"Loss: {report['initial_train_loss']:.6g} -> {report['final_train_loss']:.6g}; validation accuracy: {report['validation_accuracy']:.3f}"
    )
    print(f"Strict numerical acceptance: {report['strict_numerical_acceptance']}")
    raise SystemExit(0 if report["strict_numerical_acceptance"] else 1)
