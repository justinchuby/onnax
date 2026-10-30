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


def audit(model, reference, batch, *, sync=True):
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
    expected = reference(**inputs, use_cache=False).logits
    expected_loss = token_loss(expected, batch["labels"])
    valid = batch["attention_mask"].bool()
    np.testing.assert_allclose(
        actual.logits.detach()[valid], expected.detach()[valid], rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        actual.loss.detach(), expected_loss.detach(), rtol=1e-4, atol=1e-6
    )
    actual.loss.backward()
    expected_loss.backward()
    maximum = 0.0
    for row in model.mapping:
        layer = reference.get_submodule(row["module"])
        for key in ("A", "B"):
            index = model.onnx.names.index(row[key])
            gradient = model.onnx.adapters[index].grad.numpy()
            expected_gradient = getattr(layer, key).grad.numpy()
            maximum = max(maximum, float(np.max(np.abs(gradient - expected_gradient))))
            np.testing.assert_allclose(
                gradient, expected_gradient, rtol=1e-4, atol=1e-6, err_msg=row[key]
            )
    return {
        "logits_max_abs": float((actual.logits[valid] - expected[valid]).detach().abs().max()),
        "gradients_max_abs": maximum,
        "loss": float(actual.loss.detach()),
    }
