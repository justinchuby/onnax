# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify the public bridge on Mobius Qwen3 and independent Transformers updates."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.sft import prepare, token_loss
from examples.huggingface.validation import add_reference_adapters
from onnax.torch import OnnxModule


def test_public_module_qwen_updates_and_export(tmp_path):
    torch.set_num_threads(1)
    task, reference = prepare()
    reference = add_reference_adapters(reference, task)
    params = task.onnx.arrays()
    source = task.onnx.model
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    feeds = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
    }
    labels = ids.masked_fill(mask == 0, -100)
    frozen = {
        n: p.detach().clone() for n, p in reference.named_parameters() if not p.requires_grad
    }
    with OnnxModule(source, params, outputs=["logits"]) as model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
        target_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
        )
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            target_optimizer.zero_grad()
            actual = model(**feeds)
            expected = reference(**feeds).logits
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            loss = token_loss(actual, labels)
            target_loss = token_loss(expected, labels)
            torch.testing.assert_close(loss, target_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            target_loss.backward()
            losses.append(loss.item())
            for row in task.mapping:
                layer = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    torch.testing.assert_close(
                        model.parameters_by_name()[row[key]].grad,
                        getattr(layer, key).grad,
                        rtol=1e-4,
                        atol=1e-6,
                    )
            optimizer.step()
            target_optimizer.step()
            for row in task.mapping:
                layer = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    torch.testing.assert_close(
                        model.parameters_by_name()[row[key]],
                        getattr(layer, key),
                        rtol=1e-4,
                        atol=1e-6,
                    )
        assert losses[-1] < losses[0]
        expected = model(**feeds).detach()
        model.export(tmp_path / "qwen")
    with OnnxModule.load(tmp_path / "qwen") as loaded:
        torch.testing.assert_close(loaded(**feeds), expected, rtol=0, atol=0)
    for name, tensor in reference.named_parameters():
        if name in frozen:
            np.testing.assert_array_equal(tensor.detach().numpy(), frozen[name].numpy())
