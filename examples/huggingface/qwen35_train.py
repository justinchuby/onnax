# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify tiny hybrid Qwen3.5 LoRA with an external FP32 AdamW loop."""

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version

import onnx
import onnxruntime as ort
import torch
from onnxscript import optimizer

from examples.huggingface.qwen35 import prepare_graph
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from onnax.torch import OnnxModule


def batch(*, validation=False):
    """A deterministic next-token pattern with separate validation sequences."""
    start = 4 if validation else 3
    ids = torch.tensor(
        [[1, *range(start, start + 10), 2], [1, *range(start + 2, start + 12), 2]]
    )
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(ids.shape[1])[None].expand_as(ids),
    }


def _frozen(model):
    return {
        name: hashlib.sha256(value.const_value.numpy().tobytes()).hexdigest()
        for name, value in model.graph.initializers.items()
    }


def run(directory, *, steps=100, resume=None):
    """Align logits, token loss and every adapter gradient against Transformers."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("Use at least two updates")
    os.makedirs(directory, exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    graph, arrays, mapping, reference = prepare_graph()
    frozen = _frozen(graph)
    reference = add_reference_parameters(reference, arrays, mapping)
    audit_reference = copy.deepcopy(reference)
    frozen_reference = {
        name: parameter.detach().numpy().tobytes()
        for name, parameter in reference.named_parameters()
        if not parameter.requires_grad
    }
    train, validation = batch(), batch(validation=True)
    data_identity = {
        "version": 1,
        "config": reference.config.to_dict(),
        "mapping": mapping,
        "train_ids": train["input_ids"].tolist(),
        "validation_ids": validation["input_ids"].tolist(),
        "qk_l2norm_eps": 1e-6,
    }
    data_hash = hashlib.sha256(json.dumps(data_identity, sort_keys=True).encode()).hexdigest()
    losses, maximum_gradient_error, maximum_parameter_error = [], 0.0, 0.0
    trajectory_diagnostics = []
    with OnnxModule(
        graph, arrays, outputs=["logits"], input_gradients=[], optimize=optimizer.optimize
    ) as model:
        update = torch.optim.AdamW(model.parameters(), lr=0.03, eps=1e-5, weight_decay=0)
        reference_update = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad],
            lr=0.03,
            eps=1e-5,
            weight_decay=0,
        )
        start_step = 0
        if resume is not None:
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if state["data_identity"] != data_hash:
                raise ValueError("Checkpoint model, adapter mapping or training data changed")
            model.load_state_dict(state["model"])
            update.load_state_dict(state["optimizer"])
            reference.load_state_dict(state["reference"])
            reference_update.load_state_dict(state["reference_optimizer"])
            start_step = state["step"]
            if not 0 < start_step < steps:
                raise ValueError("Resume step must precede the requested final step")
        with torch.no_grad():
            validation_before = float(token_loss(model(**validation), validation["input_ids"]))
        for step in range(start_step, steps):
            update.zero_grad(set_to_none=True)
            reference_update.zero_grad(set_to_none=True)
            audit_reference.zero_grad(set_to_none=True)
            parameters = model.parameters_by_name()
            with torch.no_grad():
                for row in mapping:
                    target = audit_reference.get_submodule(row["module"])
                    for key in ("A", "B"):
                        getattr(target, key).copy_(parameters[row[key]])
            logits = model(**train)
            expected = audit_reference(**train, use_cache=False).logits
            independent_logits = reference(**train, use_cache=False).logits
            loss = token_loss(logits, train["input_ids"])
            expected_loss = token_loss(expected, train["input_ids"])
            independent_loss = token_loss(independent_logits, train["input_ids"])
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            expected_loss.backward()
            independent_loss.backward()
            trajectory_diagnostics.append(
                {
                    "step": step,
                    "loss": float(independent_loss.detach()),
                    "logit_max_absolute_error": float(
                        (logits.detach() - independent_logits.detach()).abs().max()
                    ),
                    "logit_pointwise_violations": int(
                        (
                            ~torch.isclose(
                                logits.detach(),
                                independent_logits.detach(),
                                rtol=1e-4,
                                atol=1e-6,
                            )
                        ).sum()
                    ),
                    "parameter_pointwise_violations": 0,
                }
            )
            for row in mapping:
                source = audit_reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    parameter, target = parameters[row[key]], getattr(source, key)
                    torch.testing.assert_close(
                        parameter.grad, target.grad, rtol=1e-4, atol=1e-6
                    )
                    maximum_gradient_error = max(
                        maximum_gradient_error,
                        float((parameter.grad - target.grad).abs().max()),
                    )
                    if step == 0:
                        assert bool(parameter.grad.count_nonzero()) == (key == "B")
                    if step == 1 and key == "A":
                        assert parameter.grad.count_nonzero() > 0
            update.step()
            reference_update.step()
            for row in mapping:
                source = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    parameter, target = parameters[row[key]], getattr(source, key)
                    # Compare independently accumulated updates in tensor norm;
                    # retain the maximum coordinate error in the report.
                    error = torch.linalg.vector_norm(parameter.detach() - target.detach())
                    bound = 1e-6 + 1e-4 * torch.linalg.vector_norm(target.detach())
                    if error > bound:
                        raise AssertionError(
                            f"Independent parameter trajectory diverged: {row[key]}"
                        )
                    trajectory_diagnostics[-1]["parameter_pointwise_violations"] += int(
                        (
                            ~torch.isclose(
                                parameter.detach(), target.detach(), rtol=1e-4, atol=1e-6
                            )
                        ).sum()
                    )
                    maximum_parameter_error = max(
                        maximum_parameter_error,
                        float((parameter.detach() - target.detach()).abs().max()),
                    )
            losses.append(float(loss.detach()))
        with torch.no_grad():
            prediction = model(**validation)
            train_after = float(token_loss(model(**train), train["input_ids"]))
            validation_after = float(token_loss(prediction, validation["input_ids"]))
            reference_train_after = float(
                token_loss(reference(**train, use_cache=False).logits, train["input_ids"])
            )
            reference_validation_after = float(
                token_loss(
                    reference(**validation, use_cache=False).logits, validation["input_ids"]
                )
            )
            torch.testing.assert_close(
                torch.tensor([train_after, validation_after]),
                torch.tensor([reference_train_after, reference_validation_after]),
                rtol=1e-4,
                atol=1e-6,
            )
        exported = os.path.join(directory, "export")
        bound = model.export(exported)
        if any(_frozen(bound).get(name) != digest for name, digest in frozen.items()):
            raise AssertionError("Frozen ONNX base weights changed")
        for name, parameter in reference.named_parameters():
            if (
                not parameter.requires_grad
                and parameter.detach().numpy().tobytes() != frozen_reference[name]
            ):
                raise AssertionError("Frozen Transformers base weights changed")
        with OnnxModule.load(exported) as restored:
            torch.testing.assert_close(restored(**validation), prediction, rtol=0, atol=0)
        for name in ("model.onnx", "execution.onnx", "derivative.onnx", "forward.onnx"):
            onnx.checker.check_model(os.path.join(exported, name), full_check=True)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            os.path.join(exported, "forward.onnx"), options, providers=["CPUExecutionProvider"]
        )
        bound_prediction = session.run(
            None, {name: value.numpy() for name, value in validation.items()}
        )[0]
        torch.testing.assert_close(
            torch.from_numpy(bound_prediction), prediction, rtol=1e-4, atol=1e-6
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": update.state_dict(),
                "reference": reference.state_dict(),
                "reference_optimizer": reference_update.state_dict(),
                "step": steps,
                "data_identity": data_hash,
            },
            os.path.join(directory, "checkpoint.pt"),
        )
    if train_after >= losses[0] or validation_after >= validation_before:
        raise AssertionError("Training and validation token loss must decrease")
    report = {
        "model": "offline-tiny-qwen3.5",
        "steps": steps,
        "start_step": start_step,
        "dtype": "float32",
        "adapter_tensors": len(arrays),
        "mapping": mapping,
        "train_loss": [losses[0], train_after],
        "validation_loss": [validation_before, validation_after],
        "reference_final_loss": {
            "train": reference_train_after,
            "validation": reference_validation_after,
        },
        "maximum_gradient_absolute_error": maximum_gradient_error,
        "maximum_parameter_absolute_error": maximum_parameter_error,
        "frozen_weights_unchanged": True,
        "export_reload_bitwise": True,
        "elapsed_seconds": time.perf_counter() - start,
        "elapsed_scope": "graph construction, reference alignment, training, checker and reload",
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "process and runtimes, three tiny model copies, reference autograd and export",
        "model_node_count": len(graph.graph),
        "trainable_parameter_bytes": sum(value.nbytes for value in arrays.values()),
        "bound_forward_loaded_and_executed": True,
        "losses": losses,
        "trajectory_diagnostics": trajectory_diagnostics,
        "parameter_comparison": "L2(error) <= 1e-6 + 1e-4 * L2(reference)",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("torch", "onnxruntime", "onnxscript", "mobius-onnx", "transformers")
        },
        "data_identity": data_hash,
        "threads": {"torch": torch.get_num_threads(), "ort_intra": 1, "ort_inter": 1},
    }
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, steps=args.steps, resume=args.resume)))


if __name__ == "__main__":
    main()
