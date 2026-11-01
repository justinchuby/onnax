# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train the MLP through the public bridge: python -m examples.torch_mlp."""

import argparse
import json
import os

import numpy as np
import torch

from examples.mlp import NAMES, SHAPES, dataset, model
from onnax.torch import OnnxModule


def run(directory, *, steps=50):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    x, y = dataset(rng, 384)
    vx, vy = dataset(rng, 192)
    initial = {
        n: np.zeros(s, np.float32) if "bias" in n else rng.normal(0, 0.2, s).astype(np.float32)
        for n, s in zip(NAMES, SHAPES, strict=True)
    }
    reference = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.tensor(initial[n])) for n in NAMES]
    )
    target_optimizer = torch.optim.AdamW(reference, lr=0.03, weight_decay=0)
    features, targets = torch.tensor(x), torch.tensor(y)
    with OnnxModule(model(), initial, input_gradients=()) as module:
        update = torch.optim.AdamW(module.parameters(), lr=0.03, weight_decay=0)
        losses = []
        for _ in range(steps):
            update.zero_grad()
            target_optimizer.zero_grad()
            logits, loss = module(features=features, targets=targets)
            w1, b1, w2, b2 = reference
            expected = torch.tanh(features @ w1 + b1) @ w2 + b2
            expected_loss = -(targets * torch.log_softmax(expected, -1)).sum(-1).mean()
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            expected_loss.backward()
            for actual, target in zip(module.trainable, reference, strict=True):
                torch.testing.assert_close(actual.grad, target.grad, rtol=1e-4, atol=1e-6)
            update.step()
            target_optimizer.step()
            for actual, target in zip(module.trainable, reference, strict=True):
                torch.testing.assert_close(actual, target, rtol=1e-4, atol=1e-6)
            losses.append(loss.item())
        validation_logits, validation_loss = module(
            features=torch.tensor(vx), targets=torch.tensor(vy)
        )
        accuracy = (
            (validation_logits.argmax(-1) == torch.tensor(vy).argmax(-1)).float().mean().item()
        )
        assert losses[-1] < losses[0] * 0.5
        assert accuracy >= 0.95
        bundle = os.path.join(directory, "bundle")
        module.export(bundle)
    with OnnxModule.load(bundle) as restored:
        logits, loss = restored(features=torch.tensor(vx), targets=torch.tensor(vy))
        torch.testing.assert_close(logits, validation_logits, rtol=0, atol=0)
        torch.testing.assert_close(loss, validation_loss, rtol=0, atol=0)
    report = {
        "steps": steps,
        "train_loss": [losses[0], losses[-1]],
        "validation_loss": validation_loss.item(),
        "validation_accuracy": accuracy,
        "independent_updates": True,
        "reload": True,
    }
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="torch_mlp_models")
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
