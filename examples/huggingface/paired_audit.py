# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Separate Qwen primal differences from VJP runtime differences on native MLX."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time

import numpy as np
import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.layer_audit import compare
from examples.huggingface.qwen import MODEL_ID, REVISION
from examples.huggingface.residuals import split_vjp
from examples.huggingface.segmented import SegmentedSession
from examples.huggingface.sft import prepare, token_loss
from examples.huggingface.validation import add_reference_adapters, audit
from examples.lora.mlx_runtime import register_library, session


def comparisons(names, actual, expected):
    rows = []
    for name, value, reference in zip(names, actual, expected, strict=True):
        if not np.isfinite(value).all() or not np.isfinite(reference).all():
            raise AssertionError(f"Nonfinite gradient: {name}")
        rows.append({"name": name, "shape": list(value.shape), **compare(value, reference)})
    return {
        "tensors": len(rows),
        "failed_tensors": sum(row["violations"] > 0 for row in rows),
        "violations": sum(row["violations"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "rows": rows,
    }


def run(*, real=False, library=None, target_nodes=1024):
    library_hash = None
    if library is not None:
        with open(library, "rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        register_library(library)
    torch.set_num_threads(1)
    model, reference = prepare(real, nf4=True, save_activations=True)
    runtime = model.onnx
    forward, _, _ = split_vjp(runtime.model, runtime.vjp_model)
    native_forward = SegmentedSession(
        forward, target_nodes=target_nodes, session_factory=session
    )
    native_backward = SegmentedSession(
        runtime.optimized_vjp, target_nodes=target_nodes, session_factory=session
    )
    if real:
        ids = torch.arange(3, 67).reshape(1, 64)
    else:
        ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    data = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
    data["position_ids"] = np.maximum(np.cumsum(mask.numpy(), axis=-1) - 1, 0)
    parameters = runtime.arrays()
    supplied = {**data, **parameters}
    names = list(dict.fromkeys([runtime.output, *runtime.residual_names]))
    states, losses, execution = {}, {}, {}
    for name, executor in (("cpu", runtime.forward_session), ("mlx", native_forward)):
        values = dict(zip(names, executor.run(names, supplied), strict=True))
        logits = torch.tensor(values[runtime.output], requires_grad=True)
        loss = token_loss(logits, labels)
        loss.backward()
        states[name] = {**supplied, **values, runtime.seed: logits.grad.numpy()}
        losses[name] = float(loss.detach())
    results = {}
    for primal, state in states.items():
        outputs = {}
        for backend, executor in (("cpu", runtime.backward_session), ("mlx", native_backward)):
            inputs = {
                name: state[name]
                for name in (v.name for v in runtime.optimized_vjp.graph.inputs)
            }
            start = time.perf_counter()
            outputs[backend] = executor.run(runtime.gradients, inputs)
            execution[f"{primal}_primal_{backend}_vjp"] = time.perf_counter() - start
        results[primal] = outputs
    same_primal = {
        name: comparisons(runtime.names, result["mlx"], result["cpu"])
        for name, result in results.items()
    }
    primal_effect = comparisons(runtime.names, results["mlx"]["cpu"], results["cpu"]["cpu"])
    reference = add_reference_adapters(reference, model)
    independent_cpu = audit(model, reference, batch, sync=False, diagnose_rounding=True)
    runtime.forward_session, runtime.backward_session = native_forward, native_backward
    independent = audit(model, reference, batch, sync=False, diagnose_rounding=True)
    if frozen_hashes(runtime.model) != runtime.frozen:
        raise AssertionError("Frozen ONNX buffers changed")
    return {
        "model": MODEL_ID if real else "tiny-qwen3-nf4",
        "revision": REVISION if real else None,
        "quantization": "onnax-row-major-nf4-v1",
        "input_ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "dtype": "float32",
        "rtol": 1e-4,
        "atol": 1e-6,
        "provider": "MLXExecutionProvider",
        "cpu_fallback_disabled": True,
        "ort_graph_optimization": "disabled",
        "same_primal_vjp": same_primal,
        "primal_swap_cpu_vjp": primal_effect,
        "independent_transformers": independent,
        "independent_transformers_cpu": independent_cpu,
        "losses": losses,
        "segment_target_nodes": target_nodes,
        "segments": {
            "forward": len(native_forward.segments),
            "vjp": len(native_backward.segments),
        },
        "construction_seconds": {
            "forward": native_forward.construction_seconds,
            "vjp": native_backward.construction_seconds,
        },
        "execution_seconds": execution,
        "timing_scope": "One execution per primal/backend pair after construction; CPU dispatch, GPU completion and NumPy exchange",
        "logical_vjp_boundary_bytes": native_backward.last_boundary_bytes,
        "boundary": "CPU PyTorch loss and parameters; NumPy arrays between sessions",
        "frozen_buffers_unchanged": True,
        "library": os.path.abspath(library) if library else "installed plugin",
        "library_sha256": library_hash,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "onnxruntime",
                "onnxruntime-ep-mlx",
                "onnx",
                "onnxscript",
                "onnx-ir",
                "torch",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--segment-nodes", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(real=args.real, library=args.library, target_nodes=args.segment_nodes)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "same_primal_vjp": {
                    k: v["failed_tensors"] for k, v in result["same_primal_vjp"].items()
                },
                "primal_swap": result["primal_swap_cpu_vjp"]["failed_tensors"],
            }
        )
    )


if __name__ == "__main__":
    main()
