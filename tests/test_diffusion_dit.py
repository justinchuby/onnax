# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent denoiser forward/VJP, dynamic spatial shapes and saved execution."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import numpy as np
import pytest

pytest.importorskip("mobius")
torch = pytest.importorskip("torch")
from onnxscript import optimizer

from examples.diffusion.dit import build, inject
from examples.diffusion.reference import forward
from examples.diffusion.train import run
from onnax.torch import OnnxModule


@pytest.mark.parametrize("shape,text_length", [((1, 4, 8, 12), 3), ((3, 4, 12, 8), 7)])
def test_conditional_denoiser_dynamic_forward_and_adapter_gradients(shape, text_length):
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    config, base, weights = build()
    graph, params, mapping = inject(base)
    rng = np.random.default_rng(71)
    for name in params:
        if name.endswith(".B"):
            params[name] = rng.normal(0, 0.02, params[name].shape).astype(np.float32)
    reference_params = {
        name: torch.tensor(value, requires_grad=True) for name, value in params.items()
    }
    batch = {
        "sample": torch.randn(shape),
        "timestep": torch.arange(shape[0], dtype=torch.float32) * 37,
        "encoder_hidden_states": torch.randn(shape[0], text_length, 32),
    }
    expected = forward(
        config,
        {k: torch.from_numpy(v) for k, v in weights.items()},
        reference_params,
        mapping,
        **batch,
    )
    target = torch.randn(shape)
    with OnnxModule(graph, params, input_gradients=[], optimize=optimizer.optimize) as model:
        actual = model(**batch)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        (actual - target).square().mean().backward()
        (expected - target).square().mean().backward()
        for name, parameter in model.parameters_by_name().items():
            torch.testing.assert_close(
                parameter.grad, reference_params[name].grad, rtol=1e-4, atol=1e-6
            )
        assert len(model.parameters_by_name()) == 16
        assert all(p.grad.abs().sum() > 0 for p in model.parameters())


def test_external_updates_frozen_base_and_loaded_denoiser(tmp_path):
    report = run(tmp_path / "dit", steps=8)
    assert report["training_loss"][1] < report["training_loss"][0]
    assert report["adapter_gradients_aligned"] and report["independent_updates_norm_aligned"]
    assert report["frozen_weights_unchanged"] and report["export_load_bitwise_equal"]
    assert len(report["adapter_mapping"]) == 8
    assert report["bound_forward_loaded_and_executed"]


def test_adapter_names_cannot_be_registered_twice():
    _, base, _ = build()
    graph, _, _ = inject(base)
    with pytest.raises(ValueError, match="collision"):
        inject(graph)


def test_fresh_process_optimizer_recovery(tmp_path):
    import os
    import subprocess
    import sys

    full = tmp_path / "full"
    first = tmp_path / "first"
    resumed = tmp_path / "resumed"
    run(full, steps=8)
    run(first, steps=4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.diffusion.train",
            "--output-dir",
            str(resumed),
            "--steps",
            "4",
            "--resume",
            str(first / "checkpoint.pt"),
        ],
        env=os.environ.copy(),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)

    def equal(left, right):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                equal(a, b)
        else:
            assert left == right

    equal(expected, actual)


def test_resume_rejects_changed_denoising_data(tmp_path):
    source = tmp_path / "source"
    run(source, steps=2)
    state = torch.load(source / "checkpoint.pt", weights_only=True)
    state["data"]["noise"] = "changed"
    altered = tmp_path / "changed.pt"
    torch.save(state, altered)
    with pytest.raises(ValueError, match="denoising data changed"):
        run(tmp_path / "resumed", steps=2, resume=altered)
