# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare saved VJP layout boundaries in isolated CPU runtime processes."""

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import onnax
from examples.huggingface.layer_audit import compare
from examples.huggingface.nf4 import lower_decoder
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.qwen import REVISION, build_base, decompose, inject, reference_model
from examples.huggingface.residuals import split_vjp
from examples.huggingface.rms_audit import diagnostic_session
from examples.huggingface.serialization import save_graph
from examples.huggingface.sft import token_loss


def worker(directory, variant, *, warmup, samples):
    torch.set_num_threads(1)
    with np.load(os.path.join(directory, "inputs.npz"), allow_pickle=False) as stored:
        inputs = dict(stored)
    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    start = time.perf_counter()
    forward = diagnostic_session(os.path.join(directory, variant + "-forward.onnx"))
    backward = diagnostic_session(os.path.join(directory, variant + "-backward.onnx"))
    construction = time.perf_counter() - start
    cases = []
    for index, b in enumerate((0.0, 0.01)):
        arguments = {
            name: np.full_like(value, b) if name.endswith(".lora.B") else value
            for name, value in inputs.items()
        }
        values = dict(
            zip(
                [value.name for value in forward.get_outputs()],
                forward.run(None, arguments),
                strict=True,
            )
        )
        logits = torch.tensor(values["logits"], requires_grad=True)
        loss = token_loss(logits, torch.from_numpy(inputs["input_ids"]))
        loss.backward()
        available = {**arguments, **values, interface["seed"]: logits.grad.numpy()}
        feeds = {value.name: available[value.name] for value in backward.get_inputs()}
        gradients = backward.run(interface["gradients"], feeds)
        arrays = dict(zip(interface["gradients"], gradients, strict=True))
        if not all(np.isfinite(value).all() for value in (*arrays.values(), values["logits"])):
            raise AssertionError("Nonfinite logits or gradients after layout splitting")
        np.savez(
            os.path.join(directory, f"{variant}-{index}.npz"),
            logits=values["logits"],
            **arrays,
        )
        residual_names = [
            name for name in feeds if name not in arguments and name != interface["seed"]
        ]
        cases.append(
            {
                "adapter_b": b,
                "loss": float(loss.detach()),
                "logical_residual_bytes": sum(values[name].nbytes for name in residual_names),
            }
        )
    for _ in range(warmup):
        backward.run(interface["gradients"], feeds)
    timings = []
    for index in range(samples):
        start = time.perf_counter()
        backward.run(interface["gradients"], feeds)
        timings.append(time.perf_counter() - start)
        if (index + 1) % 10 == 0:
            print(json.dumps({"variant": variant, "measured": index + 1}), flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "construction_seconds": construction,
        "process_peak_rss_bytes": int(rss * (1 if sys.platform == "darwin" else 1024)),
        "cases": cases,
        "backward_seconds": {
            "warmup": warmup,
            "samples": samples,
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
        },
    }
    with open(os.path.join(directory, variant + ".json"), "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def run(*, real=False, nf4=False, warmup=10, samples=100):
    if warmup < 0 or samples < 1:
        raise ValueError("Expected nonnegative warmup and positive sample counts")
    torch.set_num_threads(1)
    start = time.perf_counter()
    reference = reference_model(real)
    model, params, _ = inject(build_base(reference))
    model = decompose(model)
    if nf4:
        model, _ = lower_decoder(model, reference)
    del reference
    model_build_seconds = time.perf_counter() - start
    print(
        json.dumps({"phase": "differentiate", "model_build_seconds": model_build_seconds}),
        flush=True,
    )
    start = time.perf_counter()
    joint = onnax.vjp(model, inputs=list(params), outputs=["logits"])
    derivative_build_seconds = time.perf_counter() - start
    print(
        json.dumps({"phase": "split", "derivative_build_seconds": derivative_build_seconds}),
        flush=True,
    )
    interface = {
        "seed": joint.graph.inputs[-1].name,
        "gradients": [value.name for value in joint.graph.outputs[len(model.graph.outputs) :]],
    }
    variants, checks = {}, []
    with tempfile.TemporaryDirectory(prefix="onnax-residual-audit-") as directory:
        ids = np.arange(3, 67, dtype=np.int64)[None]
        np.savez(
            os.path.join(directory, "inputs.npz"),
            input_ids=ids,
            attention_mask=np.ones_like(ids),
            position_ids=np.arange(64, dtype=np.int64)[None],
            **params,
        )
        with open(os.path.join(directory, "interface.json"), "w") as stream:
            json.dump(interface, stream)
        for variant, enabled in (("baseline", False), ("candidate", True)):
            forward, backward, names = split_vjp(model, joint, optimize_layout=enabled)
            backward = optimize_training_graph(backward)
            payload = 0
            for kind, graph in (("forward", forward), ("backward", backward)):
                stem = variant + "-" + kind
                path = os.path.join(directory, stem + ".onnx")
                save_graph(graph, path, data_name=stem + ".data")
                payload += os.path.getsize(path) + os.path.getsize(
                    os.path.join(directory, stem + ".data")
                )
            variants[variant] = {
                "residuals": len(names),
                "forward_nodes": len(forward.graph),
                "backward_nodes": len(backward.graph),
                "backward_initializer_bytes": sum(
                    int(value.const_value.nbytes)
                    for value in backward.graph.initializers.values()
                ),
                "exported_bytes": payload,
            }
            log_path = os.path.join(directory, variant + ".log")
            print(json.dumps({"variant": variant, "worker_log": log_path}), flush=True)
            with open(log_path, "w") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "examples.huggingface.residual_audit",
                        "--worker",
                        variant,
                        "--directory",
                        directory,
                        "--warmup",
                        str(warmup),
                        "--samples",
                        str(samples),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                with open(log_path) as stream:
                    raise RuntimeError(stream.read()[-4000:])
            with open(os.path.join(directory, variant + ".json")) as stream:
                variants[variant].update(json.load(stream))
        for index in range(2):
            with (
                np.load(
                    os.path.join(directory, f"baseline-{index}.npz"), allow_pickle=False
                ) as old,
                np.load(
                    os.path.join(directory, f"candidate-{index}.npz"), allow_pickle=False
                ) as new,
            ):
                gradients = [
                    {"name": name, **compare(new[name], old[name])}
                    for name in interface["gradients"]
                ]
                checks.append(
                    {
                        "adapter_b": variants["baseline"]["cases"][index]["adapter_b"],
                        "logits": compare(new["logits"], old["logits"]),
                        "loss": compare(
                            np.asarray(variants["candidate"]["cases"][index]["loss"]),
                            np.asarray(variants["baseline"]["cases"][index]["loss"]),
                        ),
                        "failed_gradient_tensors": sum(
                            value["violations"] > 0 for value in gradients
                        ),
                        "gradients": gradients,
                    }
                )
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "revision": REVISION if real else None,
        "nf4": nf4,
        "model_build_seconds": model_build_seconds,
        "derivative_build_seconds": derivative_build_seconds,
        "variants": variants,
        "numerical_checks": checks,
        "timing_scope": "VJP execution on fixed saved FP32 residuals and seed; excludes forward, loss and parameter updates",
        "rss_scope": "isolated process including imports, runtime sessions, two correctness states and measured VJP calls",
        "boundary_scope": "logical NumPy residual sizes; shared storage may be counted more than once",
        "ort_graph_optimization": "disabled",
        "ort_memory_reuse": False,
        "threads": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnax", "onnxruntime", "torch", "onnxscript", "onnx-ir")
        },
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--directory")
    args = parser.parse_args()
    if args.worker:
        worker(args.directory, args.worker, warmup=args.warmup, samples=args.samples)
        return
    if args.output is None:
        parser.error("--output is required")
    report = run(real=args.real, nf4=args.nf4, warmup=args.warmup, samples=args.samples)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if any(
        case["failed_gradient_tensors"]
        or case["logits"]["violations"]
        or case["loss"]["violations"]
        for case in report["numerical_checks"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
