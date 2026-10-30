# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional CPU autograd and Trainer integration regressions."""

# ruff: file-ignore[module-import-not-at-top-of-file] - Skip optional dependencies before importing the integration.

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")
torch = pytest.importorskip("torch")

from examples.huggingface.bridge import OnnxModule
from examples.huggingface.trainer import run
from examples.lora.model import PARAMETERS, adapt, base_model
from examples.lora.train import problem, reference


def test_bridge_upstream_accumulation_and_snapshot():
    w, c, x, y, _, _, params = problem()
    model = OnnxModule(adapt(base_model(w, c)), params)
    for n in params:
        params[n] += np.float32(0.02)
    with torch.no_grad():
        for p, n in zip(model.adapters, PARAMETERS, strict=True):
            p.copy_(torch.from_numpy(params[n]))
    batch = torch.tensor(x[:7])
    target = torch.tensor(y[:7])
    prediction = model(features=batch)
    # A caller reusing its input buffer must not change saved backward inputs.
    batch.fill_(0)
    (3 * (prediction - target).square().mean()).backward()
    expected = reference(x[:7], y[:7], params, w, c)
    np.testing.assert_allclose(prediction.detach(), expected[0], rtol=1e-4, atol=1e-6)
    for n, p in zip(PARAMETERS, model.adapters, strict=True):
        np.testing.assert_allclose(p.grad, 3 * expected[2][n], rtol=1e-4, atol=1e-6)
    (2 * (model(features=torch.tensor(x[:7])) - target).square().mean()).backward()
    for n, p in zip(PARAMETERS, model.adapters, strict=True):
        np.testing.assert_allclose(p.grad, 5 * expected[2][n], rtol=1e-4, atol=1e-6)


def test_trainer_complete_training_and_fresh_process_resume(tmp_path):
    report = run(str(tmp_path))
    assert report["checkpoint_replay"]
