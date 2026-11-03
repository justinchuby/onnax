# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""External image preprocessing, latent scaling and benchmark comparison contracts."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional diffusion dependencies.

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("diffusers")
pytest.importorskip("mobius")
torch = pytest.importorskip("torch")
from PIL import Image

from examples.diffusion.benchmark_unet import statistics
from examples.diffusion.encode_images import encode_one, image_tensor


def test_image_rgb_normalization_and_spatial_contract(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (20, 10), (255, 0, 0)).save(path)
    actual = image_tensor(path, 16)
    assert actual.shape == (1, 3, 16, 16) and actual.dtype == torch.float32
    torch.testing.assert_close(actual[:, 0], torch.ones(1, 16, 16))
    torch.testing.assert_close(actual[:, 1:], -torch.ones(1, 2, 16, 16))
    with pytest.raises(ValueError, match="multiple of eight"):
        image_tensor(path, 15)


@pytest.mark.parametrize("masked", [True, False])
def test_external_encoding_uses_source_scaling_and_mask(masked):
    pixels = torch.ones(1, 3, 8, 8)
    latent = torch.arange(4).float().reshape(1, 4, 1, 1)
    vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=0.18215),
        encode=lambda x: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent)),
    )
    calls = []
    tokens = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 0, 0]]), attention_mask=torch.tensor([[1, 1, 0, 0]])
    )

    class Tokenizer:
        def __call__(self, caption, **kwargs):
            assert caption == "a dog" and kwargs["max_length"] == 4
            assert kwargs["padding"] == "max_length" and kwargs["truncation"]
            return tokens

    class Encoder:
        config = SimpleNamespace(max_position_embeddings=4, use_attention_mask=masked)

        def __call__(self, ids, attention_mask):
            calls.append(attention_mask)
            return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1))

    first = encode_one(pixels, "a dog", vae, Encoder(), Tokenizer(), device="cpu", seed=42)
    second = encode_one(pixels, "a dog", vae, Encoder(), Tokenizer(), device="cpu", seed=42)
    torch.testing.assert_close(first["clean"], latent * 0.18215)
    assert first["conditioning"].shape == (1, 4, 1)
    assert first["timesteps"].dtype == torch.int64
    assert all(value.device.type == "cpu" for value in first.values())
    torch.testing.assert_close(first["noise"], second["noise"], rtol=0, atol=0)
    assert (calls[0] is not None) == masked


def test_benchmark_percentiles_keep_sample_count():
    result = statistics([1.0, 2.0, 3.0])
    assert result["samples"] == 3 and result["median"] == pytest.approx(2.0)
    assert result["p95"] == pytest.approx(np.percentile([1.0, 2.0, 3.0], 95))


def test_comparison_checks_all_parameter_names_and_state(tmp_path):
    import json

    from examples.diffusion.compare_unet import compare

    common = dict.fromkeys(
        (
            "model",
            "revision",
            "export_identity",
            "data",
            "dtype",
            "latent_shape",
            "conditioning_shape",
            "learning_rate",
            "optimizer",
            "optimizer_epsilon",
        )
    )
    common.update(
        adapter_state={"adapter.A": {"shape": [2], "dtype": "float32"}},
        provider="test",
        full_step_seconds={"median": 1.0, "samples": 1},
        warmup=0,
        samples=1,
        physical_footprint={},
        peak_process_rss_bytes=0,
    )
    paths = [str(tmp_path / name) for name in ("left.json", "right.json")]
    captured = {
        "prediction": np.ones((1,), np.float32),
        "loss": np.float32(1.0),
        "adapter.A": np.ones(2, np.float32),
    }
    for path in paths:
        with open(path, "w") as stream:
            json.dump(common, stream)
        np.savez(path + ".npz", **captured)
    assert compare(*paths)["captured_tensors"] == 3
    np.savez(paths[1] + ".npz", prediction=captured["prediction"], loss=captured["loss"])
    with pytest.raises(ValueError, match="adapter names differ"):
        compare(*paths)
    changed = dict(common, export_identity="another model")
    with open(paths[1], "w") as stream:
        json.dump(changed, stream)
    with pytest.raises(ValueError, match="export_identity"):
        compare(*paths)


def test_encoded_data_detects_tensor_shape_change(tmp_path):
    import json

    from examples.diffusion.train_images import encoded_data
    from examples.huggingface.block_runtime import arrays_record

    tensors = dict(
        clean=torch.ones(1, 4, 2, 2),
        conditioning=torch.ones(1, 3, 24),
        noise=torch.zeros(1, 4, 2, 2),
        timesteps=torch.tensor([317]),
    )
    rows = []
    for split in ("train", "validation"):
        name = split + ".pt"
        torch.save(tensors, tmp_path / name)
        rows.append(
            dict(
                split=split,
                output=name,
                tensors=arrays_record({k: v.numpy() for k, v in tensors.items()}),
            )
        )
    with open(tmp_path / "manifest.json", "w") as stream:
        json.dump(dict(version=1, examples=rows), stream)
    _, splits, _ = encoded_data(tmp_path)
    assert len(splits["train"]) == len(splits["validation"]) == 1
    tensors["clean"] = tensors["clean"].reshape(1, 4, 1, 4)
    torch.save(tensors, tmp_path / "train.pt")
    with pytest.raises(ValueError, match="tensors changed"):
        encoded_data(tmp_path)


def test_benchmark_progress_preserves_completed_samples(tmp_path):
    import json

    from examples.diffusion.benchmark_unet import persist_progress

    path = str(tmp_path / "benchmark.json")
    persist_progress(
        path, warmup=2, samples=3, completed_warmup=2, records=[0.1, 0.2], latest=0.2
    )
    with open(path + ".progress.json") as stream:
        record = json.load(stream)
    assert record["completed_samples"] == 2 and record["status"] == "sampling"
    assert record["sample_seconds"] == [0.1, 0.2]
    persist_progress(
        path, warmup=2, samples=3, completed_warmup=2, records=[0.1, 0.2, 0.3], latest=0.3
    )
    with open(path + ".progress.json") as stream:
        record = json.load(stream)
    assert record["status"] == "sampling_complete" and record["completed_samples"] == 3
    assert not (tmp_path / "benchmark.json.progress.json.tmp").exists()


def test_benchmark_requires_new_output_path(tmp_path):
    from examples.diffusion.benchmark_unet import measure

    path = tmp_path / "result.json"
    path.write_text("already recorded")
    with pytest.raises(FileExistsError, match="new benchmark output"):
        measure("unused", "unused", provider="mps", output=str(path), warmup=0, samples=1)


def test_mse_validation_tracks_observed_prediction_difference():
    from examples.diffusion.train_images import validate_mse_loss

    target = torch.zeros(2)
    reference = torch.ones(2)
    actual = torch.full((2,), 1.001)
    loss = (actual - target).square().mean()
    result = validate_mse_loss(loss, actual, torch.tensor(1.0), reference, target)
    assert result["absolute_loss_difference"] <= result["mse_difference_bound"] + 1e-6
    with pytest.raises(AssertionError):
        validate_mse_loss(torch.tensor(0.5), actual, torch.tensor(1.0), reference, target)


def test_mse_prediction_bound_is_tight_for_aligned_residuals():
    from examples.diffusion.precision_policy import mse_difference_bound

    assert mse_difference_bound(4.0, 0.5) == pytest.approx(2.25)
    assert pytest.approx(mse_difference_bound(4.0, 0.5)) == (2.0 + 0.5) ** 2 - 2.0**2
    with pytest.raises(ValueError, match="nonnegative"):
        mse_difference_bound(-1.0, 0.5)
