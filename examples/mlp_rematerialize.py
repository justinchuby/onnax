# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Test a hidden-preactivation recomputation candidate on the activation workload."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

from examples.activation_liveness import analyze
from examples.mlp import NAMES
from examples.mlp_bf16 import graph, reference
from examples.mlp_device import models
from examples.mlp_microbatch import Workload
from examples.mlx_memory_lifecycle import equal_arrays, state_arrays, step_arrays
from examples.mlx_provider_memory import ProviderMemory


def recompute_preactivation(model):
    """Clone the FP32 classifier's XW+b immediately before its backward consumers."""
    result = model.clone()
    analyze(result)
    nodes = list(result.graph)
    tanhs = [n for n in nodes if not n.domain and n.op_type == "Tanh"]
    if len(tanhs) != 1:
        raise ValueError("Expected one classifier Tanh")
    tanh = tanhs[0]
    z = tanh.inputs[0]
    addition = z.producer()
    if addition is None or addition.domain or addition.op_type != "Add":
        raise ValueError("Expected MatMul followed by Add before Tanh")
    product = addition.inputs[0].producer()
    if (
        product is None
        or product.domain
        or product.op_type != "MatMul"
        or z.dtype != ir.DataType.FLOAT
        or product.outputs[0].dtype != ir.DataType.FLOAT
    ):
        raise ValueError("Expected FP32 MatMul followed by Add")
    late = [n for n in nodes[nodes.index(tanh) + 1 :] if z in n.inputs]
    if not late:
        raise ValueError("Expected backward consumers of the preactivation")
    tape = ir.tape.Tape()
    new_product = tape.op("MatMul", product.inputs)
    new_product.dtype = product.outputs[0].dtype
    new_product.shape = ir.Shape(product.outputs[0].shape)
    new_z = tape.op("Add", [new_product, addition.inputs[1]])
    new_z.dtype = z.dtype
    new_z.shape = ir.Shape(z.shape)
    for node in late:
        for i, value in enumerate(node.inputs):
            if value is z:
                node.replace_input_with(i, new_z)
    result.graph.insert_before(late[0], tape.nodes)
    onnx.checker.check_model(ir.to_proto(result), full_check=True)
    return result


def worker(directory):
    import resource

    from examples.process_memory import physical_footprint

    load_before = list(os.getloadavg())
    workload = Workload(
        directory, profile="activation", precision="fp32", microbatch=16384, reload=True
    )
    params, state = workload.parameters(), None
    samples = []
    first = None
    for i in range(110):
        loss, grads, params, state, timings = workload.step(params, state)
        del loss, grads
        if i == 0:
            first = timings
        if i >= 10:
            samples.append(timings)
    return {
        "load_average_before": load_before,
        "load_average_after": list(os.getloadavg()),
        "load_scope": "OS CPU load averages over 1, 5 and 15 minutes",
        "session_creation_ms": workload.session_ms,
        "parameter_bytes": sum(p.nbytes for p in workload.initial.values()),
        "threads": {"intra_op": 1, "inter_op": 1},
        "warmup": 10,
        "samples": 100,
        "first_step_ms": first,
        "timings_ms": {
            n: {
                "median": float(np.median([s[n] for s in samples])),
                "p95": float(np.percentile([s[n] for s in samples], 95)),
            }
            for n in samples[0]
        },
        "provider_allocator": ProviderMemory().read(),
        "physical_footprint": physical_footprint(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def run(directory):
    from examples.lora.mlx_runtime import session
    from examples.lora.mlx_training import host

    source, _, _ = models("activation", "fp32")
    candidate = recompute_preactivation(source)
    analyses = {}
    for label, model in [("baseline", source), ("candidate", candidate)]:
        folder = os.path.join(directory, label)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "gradient-16384.onnx")
        ir.save(model, path)
        loaded = ir.load(path)
        analyses[label] = {
            str(alias): analyze(loaded, alias_views=alias) for alias in (False, True)
        }
    baseline = Workload(
        os.path.join(directory, "baseline"),
        profile="activation",
        precision="fp32",
        microbatch=16384,
        reload=True,
    )
    modified = Workload(
        os.path.join(directory, "candidate"),
        profile="activation",
        precision="fp32",
        microbatch=16384,
        reload=True,
    )
    params, state = baseline.parameters(), None
    other, other_state = modified.parameters(), None
    losses = []
    for _ in range(150):
        before = {n: host(p).copy() for n, p in params.items()}
        a = baseline.step(params, state)
        b = modified.step(other, other_state)
        equal_arrays(step_arrays(a), step_arrays(b))
        _, expected_loss, expected_grads = reference(
            baseline.x, baseline.y, before, precision="fp32"
        )
        np.testing.assert_allclose(host(a[0]), expected_loss, rtol=1e-4, atol=1e-6)
        for name in NAMES:
            np.testing.assert_allclose(
                host(a[1][name]), expected_grads[name], rtol=1e-4, atol=1e-6
            )
        params, state = a[2:4]
        other, other_state = b[2:4]
        losses.append(float(host(a[0])))
    if state.step != 150 or other_state.step != 150:
        raise AssertionError("Expected 150 updates")
    checkpoint = state_arrays(other, other_state)
    np.savez(os.path.join(directory, "checkpoint.npz"), **checkpoint)
    from examples.lora.adam import AdamState

    restored = Workload(
        os.path.join(directory, "candidate"),
        profile="activation",
        precision="fp32",
        microbatch=16384,
        reload=True,
    )
    with np.load(os.path.join(directory, "checkpoint.npz")) as saved:
        restored_params = {n: restored.xp.array(saved[n]) for n in NAMES}
        restored_state = AdamState(
            int(saved["adam.step"]),
            {n: restored.xp.array(saved[f"adam.first.{n}"]) for n in NAMES},
            {n: restored.xp.array(saved[f"adam.second.{n}"]) for n in NAMES},
        )
    equal_arrays(
        step_arrays(modified.step(other, other_state)),
        step_arrays(restored.step(restored_params, restored_state)),
    )
    validation = graph(
        precision="fp32",
        batch=len(baseline.vx),
        input_width=128,
        hidden_width=2048,
        classes=16,
        with_gradients=False,
    )
    ir.save(validation, os.path.join(directory, "validation.onnx"))
    values = session(ir.load(os.path.join(directory, "validation.onnx"))).run(
        None,
        {
            "features": baseline.vx,
            "targets": baseline.vy,
            **{n: host(p) for n, p in other.items()},
        },
    )
    initial_validation = reference(
        baseline.vx, baseline.vy, baseline.initial, precision="fp32"
    )[1]
    assert values[1] < initial_validation * 0.5 and losses[-1] < losses[0] * 0.5
    measurements = {}
    for label in ("baseline", "candidate"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.mlp_rematerialize",
                "--worker",
                "--output-dir",
                os.path.join(directory, label),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        measurements[label] = json.loads(completed.stdout)
    import platform
    from importlib.metadata import version

    report = {
        "version": 1,
        "profile": "activation",
        "precision": "fp32",
        "initialization_seed": 20261030,
        "steps": 150,
        "all_updates_byte_equal": True,
        "independent_gradients_checked": 150,
        "checkpoint_replay": True,
        "initial_loss": losses[0],
        "final_training_loss": losses[-1],
        "initial_validation_loss": float(initial_validation),
        "native_final_validation_loss": float(values[1]),
        "analysis": analyses,
        "measurements": measurements,
        "onnxscript_reoptimized_nodes": len(optimizer.optimize(candidate.clone()).graph),
        "provider_peak_change_bytes": measurements["candidate"]["provider_allocator"][
            "peak_bytes"
        ]
        - measurements["baseline"]["provider_allocator"]["peak_bytes"],
        "platform": platform.platform(),
        "versions": {
            n: version(n)
            for n in (
                "onnx",
                "onnx-ir",
                "onnxscript",
                "numpy",
                "mlx",
                "onnxruntime",
                "onnxruntime-ep-mlx",
            )
        },
    }
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="recompute_models")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-only",
        dest="worker",
        action="store_true",
        help="Benchmark an already exported graph in this directory",
    )
    args = parser.parse_args()
    print(json.dumps((worker if args.worker else run)(args.output_dir), indent=2))
