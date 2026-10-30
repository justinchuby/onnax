# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent PyTorch Qwen3 LoRA forward and derivative checks."""

import numpy as np
import torch

from examples.huggingface.sft import token_loss


class TorchLora(torch.nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.from_numpy(a.copy()))
        self.B = torch.nn.Parameter(torch.from_numpy(b.copy()))
        self.scale = scale

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A) @ self.B)


def add_reference_adapters(reference, model):
    for p in reference.parameters():
        p.requires_grad_(False)
    arrays = model.onnx.arrays()
    for row in model.mapping:
        parent, name = row["module"].rsplit(".", 1)
        module = reference.get_submodule(parent)
        setattr(
            module,
            name,
            TorchLora(
                getattr(module, name),
                arrays[row["A"]],
                arrays[row["B"]],
                row["alpha"] / row["rank"],
            ),
        )
    reference.eval()
    return reference


def audit(model, reference, batch, *, sync=True, diagnose_rounding=False):
    arrays = model.onnx.arrays()
    for row in model.mapping:
        layer = reference.get_submodule(row["module"])
        with torch.no_grad():
            if not sync:
                continue
            layer.A.copy_(torch.from_numpy(arrays[row["A"]]))
            layer.B.copy_(torch.from_numpy(arrays[row["B"]]))
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    inputs = {n: v for n, v in batch.items() if n != "labels"}
    actual = model(**batch)
    if actual.logits is None:
        with torch.no_grad():
            actual.logits = model(**inputs).logits
    expected = reference(**inputs, use_cache=False).logits
    expected_loss = token_loss(expected, batch["labels"])
    valid = batch["attention_mask"].bool()
    actual_values = actual.logits.detach()[valid].numpy()
    expected_values = expected.detach()[valid].numpy()
    difference = np.abs(actual_values - expected_values)
    violations = difference > 1e-6 + 1e-4 * np.abs(expected_values)
    if not diagnose_rounding:
        np.testing.assert_allclose(actual_values, expected_values, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        actual.loss.detach(), expected_loss.detach(), rtol=1e-4, atol=1e-6
    )
    actual.loss.backward()
    expected_loss.backward()
    maximum = 0.0
    gradient_failures = {}
    for row in model.mapping:
        layer = reference.get_submodule(row["module"])
        for key in ("A", "B"):
            index = model.onnx.names.index(row[key])
            gradient = model.onnx.adapters[index].grad.numpy()
            expected_gradient = getattr(layer, key).grad.numpy()
            if not np.isfinite(gradient).all() or not np.isfinite(expected_gradient).all():
                raise AssertionError(f"Nonfinite gradient: {row[key]}")
            maximum = max(maximum, float(np.max(np.abs(gradient - expected_gradient))))
            failed = np.abs(gradient - expected_gradient) > 1e-6 + 1e-4 * np.abs(
                expected_gradient
            )
            if np.any(failed):
                gradient_failures[row[key]] = {
                    "violations": int(np.count_nonzero(failed)),
                    "max_abs": float(np.max(np.abs(gradient - expected_gradient))),
                }
            if not diagnose_rounding:
                np.testing.assert_allclose(
                    gradient, expected_gradient, rtol=1e-4, atol=1e-6, err_msg=row[key]
                )

    return {
        "strict_logit_check_passed": not bool(np.any(violations)),
        "logit_violations": int(np.count_nonzero(violations)),
        "logits_relative_l2": float(
            np.linalg.norm(actual_values - expected_values) / np.linalg.norm(expected_values)
        ),
        "logits_max_abs": float((actual.logits[valid] - expected[valid]).detach().abs().max()),
        "gradients_max_abs": maximum,
        "strict_gradient_check_passed": not bool(gradient_failures),
        "gradient_failures": gradient_failures,
        "loss": float(actual.loss.detach()),
    }
