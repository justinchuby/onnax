# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Validate a pretrained UNet export on CPU or MLX without regenerating its VJP."""

import argparse
import hashlib
import json
import math
import os
import time
from importlib.metadata import version

import torch
from diffusers import DDPMScheduler

from examples.diffusion.unet import MODEL_ID, REVISION, pretrained
from examples.diffusion.unet_reference import make_reference
from examples.diffusion.unet_train import compare_prediction, synthetic_batch, training_batch
from onnax.torch import OnnxModule


def run(directory, *, provider="mlx", library=None, steps=3, batch_file=None):
    """Compare every adapter gradient and update with an independent Diffusers model."""
    if steps < 1 or provider not in ("cpu", "mlx"):
        raise ValueError("Select cpu/mlx and positive training steps")
    with open(os.path.join(directory, "report.json"), encoding="utf-8") as stream:
        source = json.load(stream)
    if source["model"] != MODEL_ID or source["revision"] != REVISION:
        raise ValueError("Expected the pinned pretrained UNet training run")
    factory = None
    if provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if library:
            register_library(library)
        factory = session
    elif library:
        raise ValueError("A development MLX library requires provider=mlx")
    torch.set_num_threads(1)
    config, weights, schedule = pretrained()
    scheduler = DDPMScheduler.from_config(schedule)
    if batch_file:
        data = torch.load(batch_file, weights_only=True, map_location="cpu")
        batch, target = training_batch(
            data["clean"], data["conditioning"], data["noise"], data["timesteps"], scheduler
        )
    else:
        batch, target = synthetic_batch(
            config, scheduler, 20261102, size=source["latent_shape"][-1]
        )
    identity = {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in {**batch, "target": target}.items()
    }
    if identity != source["data"]:
        raise ValueError("Supply the original training batch for this export")
    started = time.perf_counter()
    with OnnxModule.load(os.path.join(directory, "export"), session_factory=factory) as model:
        reference, params = make_reference(
            config, weights, model.arrays(), source["adapter_mapping"]
        )
        update = torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, weight_decay=0)
        control = torch.optim.AdamW(params.values(), lr=0.003, eps=1e-5, weight_decay=0)
        rows = []
        for step in range(steps):
            update.zero_grad(set_to_none=True)
            control.zero_grad(set_to_none=True)
            actual = model(**batch)
            expected = reference(**batch).sample
            prediction = compare_prediction(actual, expected)
            loss = (actual - target).square().mean()
            reference_loss = (expected - target).square().mean()
            torch.testing.assert_close(loss, reference_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            reference_loss.backward()
            gradient_error = 0.0
            for name, parameter in model.parameters_by_name().items():
                torch.testing.assert_close(
                    parameter.grad, params[name].grad, rtol=1e-4, atol=1e-6
                )
                gradient_error = max(
                    gradient_error, float((parameter.grad - params[name].grad).abs().max())
                )
            update.step()
            control.step()
            relative_errors = []
            for name, parameter in model.parameters_by_name().items():
                error = float(
                    torch.linalg.vector_norm(parameter.detach() - params[name].detach())
                )
                norm = float(torch.linalg.vector_norm(params[name].detach()))
                if not math.isfinite(error) or error > 1e-6 + 1e-4 * norm:
                    raise AssertionError(f"Independent parameter norm drift: {name}")
                relative_errors.append(error / max(norm, 1e-30))
            rows.append(
                dict(
                    step=step + 1,
                    loss=float(loss.detach()),
                    maximum_gradient_error=gradient_error,
                    maximum_relative_parameter_l2_error=max(relative_errors),
                    prediction=prediction,
                )
            )
    library_hash = None
    if library:
        with open(library, "rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        provider=provider,
        cpu_fallback=False,
        ort_graph_optimizer=False,
        library_sha256=library_hash,
        latent_shape=list(batch["sample"].shape),
        elapsed_including_load_and_reference_seconds=time.perf_counter() - started,
        steps=rows,
        model=MODEL_ID,
        revision=REVISION,
        versions={
            name: version(name)
            for name in ("torch", "onnxruntime", "diffusers", "mobius-onnx")
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--provider", choices=("cpu", "mlx"), default="mlx")
    parser.add_argument("--mlx-library")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        args.training_run,
        provider=args.provider,
        library=args.mlx_library,
        steps=args.steps,
        batch_file=args.batch,
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
