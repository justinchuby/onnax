# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Source UNet forward and all adapter gradients through frozen convolutions."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional example dependencies.

import os

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")
from diffusers import UNet2DConditionModel
from onnxscript import optimizer

from examples.diffusion.adapters import inject
from examples.diffusion.unet import build, tiny_config
from examples.diffusion.unet_reference import make_reference
from onnax.torch import OnnxModule


@pytest.fixture(params=["cpu", "mlx"])
def runtime_factory(request):
    if request.param == "cpu":
        return None
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library, session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    return session


@pytest.mark.parametrize("shape,text_length", [((1, 4, 8, 12), 3), ((2, 4, 12, 8), 7)])
def test_unet_gradients_and_external_updates(shape, text_length, tmp_path, runtime_factory):
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    config = tiny_config()
    source = UNet2DConditionModel(**config).eval()
    graph, params, mapping = inject(build(config, source.state_dict()))
    assert len(mapping) == 24
    rng = np.random.default_rng(61)
    for name in params:
        if name.endswith(".B"):
            params[name] = rng.normal(0, 0.02, params[name].shape).astype(np.float32)
    reference, reference_params = make_reference(config, source.state_dict(), params, mapping)
    batch = dict(
        sample=torch.randn(shape),
        timestep=torch.arange(shape[0]).float() * 13,
        encoder_hidden_states=torch.randn(shape[0], text_length, 24),
    )
    target = torch.randn(shape)
    with OnnxModule(
        graph,
        params,
        input_gradients=[],
        optimize=optimizer.optimize,
        session_factory=runtime_factory,
    ) as model:
        update = torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, weight_decay=0)
        expected_update = torch.optim.AdamW(
            reference_params.values(), lr=0.003, eps=1e-5, weight_decay=0
        )
        for _ in range(3):
            update.zero_grad()
            expected_update.zero_grad()
            actual = model(**batch)
            expected = reference(**batch).sample
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            loss = (actual - target).square().mean()
            expected_loss = (expected - target).square().mean()
            torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            expected_loss.backward()
            for name, parameter in model.parameters_by_name().items():
                torch.testing.assert_close(
                    parameter.grad, reference_params[name].grad, rtol=1e-4, atol=1e-6
                )
            update.step()
            expected_update.step()
            for name, parameter in model.parameters_by_name().items():
                torch.testing.assert_close(
                    parameter, reference_params[name], rtol=1e-4, atol=1e-6
                )
        with torch.no_grad():
            prediction = model(**batch)
        model.export(tmp_path / "trained")
        with (
            OnnxModule.load(tmp_path / "trained", session_factory=runtime_factory) as loaded,
            torch.no_grad(),
        ):
            torch.testing.assert_close(loaded(**batch), prediction, rtol=0, atol=0)


@pytest.mark.parametrize("prediction", ["epsilon", "sample", "v_prediction"])
def test_external_scheduler_training_target(prediction):
    from diffusers import DDPMScheduler

    from examples.diffusion.unet_train import training_batch

    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=prediction)
    clean = torch.arange(16).float().reshape(1, 4, 2, 2) / 10
    noise = torch.flip(clean, [1])
    timesteps = torch.tensor([317])
    condition = torch.ones(1, 3, 24)
    batch, target = training_batch(clean, condition, noise, timesteps, scheduler)
    alpha = scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
    torch.testing.assert_close(
        batch["sample"], alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    )
    expected = {
        "epsilon": noise,
        "sample": clean,
        "v_prediction": alpha.sqrt() * noise - (1 - alpha).sqrt() * clean,
    }[prediction]
    torch.testing.assert_close(target, expected)
    assert batch["timestep"].dtype == torch.float32


def test_external_scheduler_training_and_fresh_process_recovery(tmp_path):
    import subprocess
    import sys

    from examples.diffusion.unet_train import run

    full, first, resumed = (tmp_path / name for name in ("full", "first", "resumed"))
    report = run(full, steps=4)
    assert report["train_loss"][1] < report["train_loss"][0]
    assert report["external_scheduler_sampling_aligned"]
    assert report["frozen_weights_unchanged"] and report["loaded_prediction_bitwise_equal"]
    run(first, steps=2)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.diffusion.unet_train",
            "--output-dir",
            str(resumed),
            "--steps",
            "2",
            "--resume",
            str(first / "checkpoint.pt"),
        ],
        env=os.environ.copy(),
        check=True,
    )
    left = torch.load(full / "checkpoint.pt", weights_only=True)
    right = torch.load(resumed / "checkpoint.pt", weights_only=True)

    def equal(a, b):
        assert type(a) is type(b)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for name in a:
                equal(a[name], b[name])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        else:
            assert a == b

    equal(left, right)
    state = torch.load(first / "checkpoint.pt", weights_only=True)
    state["data"]["target"] = "changed"
    torch.save(state, tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="training data changed"):
        run(tmp_path / "changed", steps=1, resume=tmp_path / "changed.pt")


@pytest.mark.parametrize(
    "name,value", [("dropout", 0.1), ("downsample_padding", 0), ("act_fn", "gelu")]
)
def test_unet_rejects_configuration_not_represented_by_builder(name, value):
    config = tiny_config()
    config[name] = value
    with pytest.raises(ValueError, match=name):
        build(config, {})
