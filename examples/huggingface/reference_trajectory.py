# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent FP32 Qwen adapter updates with a calibrated head reference."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from unittest.mock import patch

import numpy as np
import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.qwen import REVISION
from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.rms_audit import diagnostic_session
from examples.huggingface.sft import prepare, texts, token_loss, tokenizer
from examples.huggingface.validation import add_reference_adapters, audit


def compare_tensors(actual, expected):
    failures, maximum, worst = {}, 0.0, None
    for name, value in actual.items():
        x, y = value.detach().numpy(), expected[name].detach().numpy()
        if x.shape != y.shape or x.dtype != y.dtype:
            raise AssertionError(f"Reference interface mismatch: {name}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f"Nonfinite reference comparison: {name}")
        difference = np.abs(x - y)
        error = float(difference.max())
        if worst is None or error > maximum:
            coordinate = tuple(
                int(i) for i in np.unravel_index(np.argmax(difference), x.shape)
            )
            worst = {
                "name": name,
                "coordinate": coordinate,
                "actual": float(x[coordinate]),
                "reference": float(y[coordinate]),
            }
            maximum = error
        failed = difference > 1e-6 + 1e-4 * np.abs(y)
        if failed.any():
            failures[name] = {"elements": int(failed.sum()), "max_abs": error}
    return {"passed": not failures, "max_abs": maximum, "failures": failures, "worst": worst}


def reference_parameters(model, reference):
    return {
        row[key]: getattr(reference.get_submodule(row["module"]), key)
        for row in model.mapping
        for key in ("A", "B")
    }


def run(*, real=False, nf4=False, steps=4, accumulation=4, eps=1e-8):
    if steps < 1 or accumulation < 1:
        raise ValueError("Expected positive step and accumulation counts")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Expected a positive finite Adam epsilon")
    torch.set_num_threads(1)
    with patch("examples.huggingface.bridge.session", diagnostic_session):
        model, reference = prepare(real, nf4=nf4, save_activations=True)
    reference = add_reference_adapters(reference, model)
    tok, corpus = tokenizer(real), texts()
    batches = []
    for line in corpus:
        encoded = tok(
            line, truncation=True, max_length=64, padding="max_length", return_tensors="pt"
        )
        batch = {name: encoded[name] for name in ("input_ids", "attention_mask")}
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        batches.append(batch)
    ordinary = audit(model, reference, batches[0], diagnose_rounding=True)
    actual_params = dict(zip(model.onnx.names, model.onnx.adapters, strict=True))
    expected_params = reference_parameters(model, reference)
    actual_optimizer = torch.optim.AdamW(
        actual_params.values(), lr=0.003, eps=eps, weight_decay=0, foreach=False
    )
    expected_optimizer = torch.optim.AdamW(
        expected_params.values(), lr=0.003, eps=eps, weight_decay=0, foreach=False
    )
    records = []
    with pairwise_head(reference):
        calibrated = audit(model, reference, batches[0], diagnose_rounding=True)
        for step in range(steps):
            actual_optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            microbatches = [
                batches[(step * accumulation + index) % len(batches)]
                for index in range(accumulation)
            ]
            tokens = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in microbatches)
            actual_loss, expected_loss = 0.0, 0.0
            for batch in microbatches:
                actual = model(**batch, num_items_in_batch=tokens).loss
                inputs = {name: value for name, value in batch.items() if name != "labels"}
                expected = token_loss(
                    reference(**inputs, use_cache=False).logits, batch["labels"], tokens
                )
                actual.backward()
                expected.backward()
                actual_loss += float(actual.detach())
                expected_loss += float(expected.detach())
            gradients = compare_tensors(
                {name: value.grad for name, value in actual_params.items()},
                {name: value.grad for name, value in expected_params.items()},
            )
            actual_optimizer.step()
            expected_optimizer.step()
            parameters = compare_tensors(actual_params, expected_params)
            states = {}
            for key in ("exp_avg", "exp_avg_sq"):
                states[key] = compare_tensors(
                    {
                        name: actual_optimizer.state[value][key]
                        for name, value in actual_params.items()
                    },
                    {
                        name: expected_optimizer.state[value][key]
                        for name, value in expected_params.items()
                    },
                )
            record = {
                "step": step + 1,
                "valid_tokens": tokens,
                "loss": actual_loss,
                "reference_loss": expected_loss,
                "gradients": gradients,
                "parameters": parameters,
                "optimizer_state": states,
            }
            worst = parameters["worst"]
            coordinate = tuple(worst["coordinate"])
            record["gradient_at_worst_parameter"] = {
                "actual": float(actual_params[worst["name"]].grad[coordinate]),
                "reference": float(expected_params[worst["name"]].grad[coordinate]),
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": actual_loss,
                        "gradient_check": gradients["passed"],
                        "parameter_check": parameters["passed"],
                    }
                ),
                flush=True,
            )
    if frozen_hashes(model.onnx.model) != model.onnx.frozen:
        raise AssertionError("Frozen graph weights changed")
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "revision": REVISION if real else None,
        "nf4": nf4,
        "dtype": "float32",
        "reference_head": {"accumulation": "pairwise", "block_size": 128},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 0.003,
            "eps": eps,
            "weight_decay": 0,
            "foreach": False,
        },
        "accumulation": accumulation,
        "parameter_synchronization": "initialization only",
        "rtol": 1e-4,
        "atol": 1e-6,
        "baseline": ordinary,
        "calibrated_initial": calibrated,
        "steps": records,
        "checks": {
            "initial_forward": calibrated["strict_logit_check_passed"],
            "initial_gradients": calibrated["strict_gradient_check_passed"],
            "loss": all(
                abs(step["loss"] - step["reference_loss"])
                <= 1e-6 + 1e-4 * abs(step["reference_loss"])
                for step in records
            ),
            "gradients": all(step["gradients"]["passed"] for step in records),
            "parameters": all(step["parameters"]["passed"] for step in records),
            "optimizer_state": all(
                state["passed"]
                for step in records
                for state in step["optimizer_state"].values()
            ),
        },
        "frozen_weights_unchanged": True,
        "corpus_sha256": hashlib.sha256("\n".join(corpus).encode()).hexdigest(),
        "ort_graph_optimization": "disabled",
        "ort_memory_reuse": False,
        "threads": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnax", "torch", "onnxruntime", "transformers", "mobius-onnx")
        },
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        real=args.real,
        nf4=args.nf4,
        steps=args.steps,
        accumulation=args.accumulation,
        eps=args.eps,
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not all(report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
