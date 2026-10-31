# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Additional stable-loss PEFT controls at the current adapter parameters."""

import numpy as np
import torch

from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.reference_losses import stable_token_loss


def _compare(actual, expected):
    failures, maximum = {}, 0.0
    for name, value in actual.items():
        x, y = value.detach().numpy(), expected[name].detach().numpy()
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid PEFT comparison tensor: {name}")
        delta = np.abs(x - y)
        maximum = max(maximum, float(delta.max()))
        failed = delta > 1e-6 + 1e-4 * np.abs(y)
        if failed.any():
            failures[name] = {"violations": int(failed.sum()), "max_abs": float(delta.max())}
    return failures, maximum


def stable_control(model, peft, batch, actual_loss):
    """Compare stable PEFT gradients and fresh Adam steps; restore PEFT gradients."""
    parameters = dict(peft.named_parameters())
    saved = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in parameters.items()
    }
    try:
        peft.zero_grad(set_to_none=True)
        inputs = {name: value for name, value in batch.items() if name != "labels"}
        bf16 = getattr(model, "projection_precision", "fp32") == "bf16"
        with pairwise_head(peft, projection_precision="bf16" if bf16 else "fp32"):
            logits = peft(**inputs, use_cache=False).logits
            loss = stable_token_loss(logits, batch["labels"])
            loss.backward()
        actual, expected = {}, {}
        actual_gradients, expected_gradients = {}, {}
        for row in model.mapping:
            for key in ("A", "B"):
                name = row[key]
                p = model.onnx.adapters[model.onnx.names.index(name)]
                q = parameters[f"base_model.model.{row['module']}.lora_{key}.default.weight"]
                actual_gradients[name] = p.grad
                expected_gradients[name] = q.grad.T
                actual[name] = torch.nn.Parameter(p.detach().clone())
                expected[name] = torch.nn.Parameter(q.detach().clone())
                actual[name].grad = p.grad.detach().clone()
                expected[name].grad = q.grad.detach().clone()
        gradient_failures, gradient_maximum = _compare(actual_gradients, expected_gradients)
        for values in (actual, expected):
            torch.optim.AdamW(values.values(), lr=0.003, weight_decay=0).step()
        update_failures, update_maximum = _compare(
            actual, {name: parameter.T for name, parameter in expected.items()}
        )
        return {
            "reference_loss": float(loss.detach()),
            "reference_computation": (
                "target-excluded stable CE; BF16 forward; FP32 pairwise head backward, K blocks 128"
                if bf16
                else "target-excluded stable CE; FP32 pairwise head backward, K blocks 128"
            ),
            "loss_check_passed": bool(
                np.isclose(
                    float(actual_loss.detach()), float(loss.detach()), rtol=1e-4, atol=1e-6
                )
            ),
            "adapter_tensors": len(actual),
            "rtol": 1e-4,
            "atol": 1e-6,
            "optimizer": {
                "name": "AdamW",
                "lr": 0.003,
                "eps": 1e-8,
                "betas": [0.9, 0.999],
                "weight_decay": 0,
            },
            "gradient_check_passed": not gradient_failures,
            "gradients_max_abs": gradient_maximum,
            "gradient_failures": gradient_failures,
            "update_check_passed": not update_failures,
            "updates_max_abs": update_maximum,
            "update_failures": update_failures,
        }
    finally:
        for name, parameter in parameters.items():
            parameter.grad = saved[name]
