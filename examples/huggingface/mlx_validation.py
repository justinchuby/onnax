# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Validate tiny NF4 Qwen forward/VJP on the native MLX execution provider."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform

import numpy as np
import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.residuals import split_vjp
from examples.huggingface.sft import prepare
from examples.huggingface.validation import add_reference_adapters, audit
from examples.lora.mlx_runtime import session


class ValidationError(AssertionError):
    """Retain the complete numerical report when a strict comparison fails."""

    def __init__(self, report):
        self.report = report
        super().__init__("Native MLX validation failed; inspect the saved per-step report")


def _reference_audit(model, reference, batch, *, stable_reference=False, **kwargs):
    if not stable_reference:
        return audit(model, reference, batch, **kwargs)
    from examples.huggingface.reference_head import pairwise_head
    from examples.huggingface.reference_losses import stable_token_loss

    with pairwise_head(reference):
        return audit(model, reference, batch, reference_loss=stable_token_loss, **kwargs)


def _same_parameter_control(model, reference, batch, after_updates, *, stable_reference=False):
    gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
    ]
    try:
        control = _reference_audit(
            model,
            copy.deepcopy(reference),
            batch,
            stable_reference=stable_reference,
            sync=True,
            diagnose_rounding=True,
        )
        return {"after_updates": after_updates, **control}
    finally:
        for parameter, gradient in zip(model.parameters(), gradients, strict=True):
            parameter.grad = gradient


def run(*, library=None, segmented=False, target_nodes=1024, steps=4, stable_reference=False):
    """Compare independent FP32 AdamW updates with CPU fallback disabled."""
    if steps < 1:
        raise ValueError("Expected a positive training step count")
    if library is not None:
        from examples.lora.mlx_runtime import register_library

        register_library(library)
    library_hash = None
    if library is not None:
        with open(library, "rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    torch.set_num_threads(1)
    model, reference = prepare(nf4=True, save_activations=True)
    forward, _, _ = split_vjp(model.onnx.model, model.onnx.vjp_model)
    if segmented:
        from examples.huggingface.segmented import SegmentedSession

        model.onnx.forward_session = SegmentedSession(
            forward, target_nodes=target_nodes, session_factory=session
        )
        model.onnx.backward_session = SegmentedSession(
            model.onnx.optimized_vjp, target_nodes=target_nodes, session_factory=session
        )
    else:
        model.onnx.forward_session = session(forward)
        model.onnx.backward_session = session(model.onnx.optimized_vjp)
    reference = add_reference_adapters(reference, model)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    independent = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    losses = []
    checks = []
    same_parameters = None
    criteria = (
        "strict_logit_check_passed",
        "strict_loss_check_passed",
        "strict_gradient_check_passed",
    )
    for step in range(steps):
        result = _reference_audit(
            model,
            reference,
            batch,
            stable_reference=stable_reference,
            sync=False,
            diagnose_rounding=True,
        )
        point = _same_parameter_control(
            model, reference, batch, step, stable_reference=stable_reference
        )
        result["same_parameter"] = point
        if same_parameters is None and not all(result[key] for key in criteria):
            same_parameters = point
        losses.append(result["loss"])
        optimizer.step()
        independent.step()
        parameters = []
        for row in model.mapping:
            for key in ("A", "B"):
                actual = model.onnx.adapters[model.onnx.names.index(row[key])].detach().numpy()
                expected = (
                    getattr(reference.get_submodule(row["module"]), key).detach().numpy()
                )
                if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                    raise FloatingPointError(f"Nonfinite updated parameter: {row[key]}")
                delta = np.abs(actual - expected)
                parameters.append(
                    {
                        "name": row[key],
                        "max_abs": float(delta.max()),
                        "violations": int(
                            np.count_nonzero(delta > 1e-6 + 1e-4 * np.abs(expected))
                        ),
                    }
                )
        result["parameters"] = parameters
        result["strict_parameter_check_passed"] = not any(
            row["violations"] for row in parameters
        )
        if same_parameters is None and not result["strict_parameter_check_passed"]:
            same_parameters = _same_parameter_control(
                model, reference, batch, step + 1, stable_reference=stable_reference
            )
        checks.append(result)
        if (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": result["loss"]}), flush=True)
    frozen = frozen_hashes(model.onnx.model) == model.onnx.frozen
    passed = (
        frozen
        and losses[-1] < losses[0]
        and all(
            all(row[key] for key in (*criteria, "strict_parameter_check_passed"))
            and all(row["same_parameter"][key] for key in criteria)
            for row in checks
        )
    )
    report = {
        "strict_checks_passed": passed,
        "shared_parameter_checks_passed": all(
            all(row["same_parameter"][key] for key in criteria) for row in checks
        ),
        "independent_updates_passed": frozen
        and losses[-1] < losses[0]
        and all(
            row["strict_parameter_check_passed"] and row["strict_loss_check_passed"]
            for row in checks
        ),
        "same_parameter_control": same_parameters,
        "model": "tiny-qwen3-nf4",
        "reference_computation": (
            "target-excluded stable CE; FP32 pairwise head backward, K blocks 128"
            if stable_reference
            else "standard PyTorch token loss and head backward"
        ),
        "execution": "segmented" if segmented else "whole_graph",
        "segment_target_nodes": target_nodes if segmented else None,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnxruntime-ep-mlx", "torch", "mlx")
        },
        "library": os.path.abspath(library) if library else "installed plugin",
        "library_sha256": library_hash,
        "checks": checks,
        "provider": "MLXExecutionProvider",
        "cpu_fallback_disabled": True,
        "ort_graph_optimization": "disabled",
        "runtime_dtype": "float32",
        "lut4_fusion_disabled": os.environ.get("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION") == "1",
        "boundary": "CPU PyTorch loss, parameters and AdamW; NumPy session exchange",
        "adapter_tensors": len(model.onnx.adapters),
        "steps": len(losses),
        "losses": losses,
        "gradient_and_update_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "frozen_buffers_unchanged": frozen,
    }
    if not passed:
        raise ValidationError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Development MLX EP dylib")
    parser.add_argument("--segmented", action="store_true")
    parser.add_argument("--segment-nodes", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--stable-reference", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    failed = False
    try:
        result = run(
            library=args.library,
            segmented=args.segmented,
            target_nodes=args.segment_nodes,
            steps=args.steps,
            stable_reference=args.stable_reference,
        )
    except ValidationError as error:
        result = error.report
        failed = True
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
