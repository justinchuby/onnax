# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train Mobius UNet adapters with an external Diffusers scheduler and AdamW."""

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import time
from importlib.metadata import version

import onnx
import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from onnxscript import optimizer

from examples.diffusion.adapters import inject
from examples.diffusion.train import frozen_hashes
from examples.diffusion.unet import MODEL_ID, REVISION, build, pretrained, tiny_config
from examples.diffusion.unet_reference import make_reference
from onnax.torch import OnnxModule


def compare_prediction(actual, expected):
    """Record FP32 prediction error and check the complete tensor's norm."""
    difference = actual.detach() - expected.detach()
    error = float(torch.linalg.vector_norm(difference))
    reference = float(torch.linalg.vector_norm(expected.detach()))
    if not math.isfinite(error) or error > 1e-6 + 1e-4 * reference:
        raise AssertionError("UNet prediction norm differs from Diffusers")
    return dict(
        max_absolute_error=float(difference.abs().max()),
        absolute_l2_error=error,
        reference_l2_norm=reference,
        pointwise_tolerance_failures=int(
            (difference.abs() > 1e-6 + 1e-4 * expected.detach().abs()).sum()
        ),
    )


def training_batch(clean, conditioning, noise, timesteps, scheduler):
    """Delegate forward diffusion and prediction targets to the external scheduler."""
    prediction = scheduler.config.prediction_type
    if prediction == "epsilon":
        target = noise
    elif prediction == "v_prediction":
        target = scheduler.get_velocity(clean, noise, timesteps)
    elif prediction == "sample":
        target = clean
    else:
        raise ValueError(f"Unsupported denoising prediction target: {prediction}")
    return dict(
        sample=scheduler.add_noise(clean, noise, timesteps),
        timestep=timesteps.float(),
        encoder_hidden_states=conditioning,
    ), target


def synthetic_batch(config, scheduler, seed, *, size=8):
    """Fixed synthetic latent/conditioning data for numerical qualification."""
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randn(1, config["in_channels"], size, size, generator=generator) * 0.3
    noise = torch.randn(clean.shape, generator=generator)
    condition = torch.randn(1, 5, config["cross_attention_dim"], generator=generator)
    timesteps = torch.tensor([317])
    return training_batch(clean, condition, noise, timesteps, scheduler)


def sample_latents(predict, configuration, initial, conditioning, *, steps=4):
    """Let the external scheduler own timesteps, scaling and solver state."""
    scheduler = DPMSolverMultistepScheduler.from_config(configuration)
    scheduler.set_timesteps(steps)
    sample = initial.clone()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            output = predict(
                sample=scheduler.scale_model_input(sample, timestep),
                timestep=timestep.float().expand(sample.shape[0]),
                encoder_hidden_states=conditioning,
            )
            sample = scheduler.step(output, timestep, sample).prev_sample
    return sample


def run(directory, *, steps=20, real=False, size=8, batch_file=None, resume=None):
    """Compare forward, loss, every adapter gradient and every external update."""
    if steps < 1 or size < 4 or size % 4:
        raise ValueError("Use positive steps and a latent size divisible by four")
    os.makedirs(directory, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    if real:
        config, weights, schedule_config = pretrained()
    else:
        config = tiny_config()
        source = UNet2DConditionModel(**config).eval()
        weights = source.state_dict()
        schedule_config = dict(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
    scheduler = DDPMScheduler.from_config(schedule_config)
    batch, target = synthetic_batch(config, scheduler, 20261102, size=size)
    validation, validation_target = synthetic_batch(config, scheduler, 20261103, size=size)
    if batch_file is not None:
        data = torch.load(batch_file, map_location="cpu", weights_only=True)
        batch, target = training_batch(
            data["clean"], data["conditioning"], data["noise"], data["timesteps"], scheduler
        )
    data_identity = {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in {**batch, "target": target}.items()
    }
    base = build(config, weights)
    graph, params, mapping = inject(base)
    frozen = frozen_hashes(graph)
    reference, expected_parameters = make_reference(config, weights, params, mapping)
    reference_update = torch.optim.AdamW(
        expected_parameters.values(), lr=0.003, eps=1e-5, weight_decay=0
    )
    started = time.perf_counter()
    losses, gradient_error, parameter_error = [], 0.0, 0.0
    parameter_diagnostics = []
    forward_diagnostics = []
    start_step = 0
    with OnnxModule(graph, params, input_gradients=[], optimize=optimizer.optimize) as model:
        construction_seconds = time.perf_counter() - started
        update = torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, weight_decay=0)
        if resume:
            checkpoint = torch.load(resume, weights_only=True, map_location="cpu")
            if checkpoint["data"] != data_identity or checkpoint["config"] != config:
                raise ValueError("Checkpoint UNet configuration or training data changed")
            model.load_state_dict(checkpoint["model"])
            update.load_state_dict(checkpoint["optimizer"])
            with torch.no_grad():
                for name, parameter in model.parameters_by_name().items():
                    expected_parameters[name].copy_(parameter)
            reference_update.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
            start_step = checkpoint["step"]
        with torch.no_grad():
            train_before = float((model(**batch) - target).square().mean())
            valid_before = float((model(**validation) - validation_target).square().mean())
        for _ in range(steps):
            update.zero_grad(set_to_none=True)
            reference_update.zero_grad(set_to_none=True)
            actual = model(**batch)
            independent = {
                name: parameter.detach().clone()
                for name, parameter in expected_parameters.items()
            }
            with torch.no_grad():
                for name, parameter in model.parameters_by_name().items():
                    expected_parameters[name].copy_(parameter)
                matched = reference(**batch).sample
                for name, value in independent.items():
                    expected_parameters[name].copy_(value)
            matched_diagnostic = compare_prediction(actual, matched)
            expected = reference(**batch).sample
            difference = (actual.detach() - expected.detach()).abs()
            forward_diagnostics.append(
                dict(
                    step=start_step + len(losses) + 1,
                    max_absolute_error=float(difference.max()),
                    rms_error=float(difference.square().mean().sqrt()),
                    pointwise_tolerance_failures=int(
                        (difference > 1e-6 + 1e-4 * expected.detach().abs()).sum()
                    ),
                    matched_parameters=matched_diagnostic,
                )
            )
            loss = (actual - target).square().mean()
            reference_loss = (expected - target).square().mean()
            torch.testing.assert_close(loss, reference_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            reference_loss.backward()
            for name, parameter in model.parameters_by_name().items():
                expected_parameter = expected_parameters[name]
                torch.testing.assert_close(
                    parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-6
                )
                gradient_error = max(
                    gradient_error,
                    float((parameter.grad - expected_parameter.grad).abs().max()),
                )
            update.step()
            reference_update.step()
            for name, parameter in model.parameters_by_name().items():
                difference = parameter.detach() - expected_parameters[name].detach()
                error_norm = float(torch.linalg.vector_norm(difference))
                reference_norm = float(
                    torch.linalg.vector_norm(expected_parameters[name].detach())
                )
                if not math.isfinite(error_norm) or error_norm > 1e-6 + 1e-4 * reference_norm:
                    raise AssertionError(f"Independent parameter norm drift: {name}")
                parameter_diagnostics.append(
                    dict(
                        step=start_step + len(losses) + 1,
                        parameter=name,
                        absolute_l2_error=error_norm,
                        reference_l2_norm=reference_norm,
                        pointwise_tolerance_failures=int(
                            (
                                difference.abs()
                                > 1e-6 + 1e-4 * expected_parameters[name].detach().abs()
                            ).sum()
                        ),
                    )
                )
                parameter_error = max(
                    parameter_error,
                    float(
                        (parameter.detach() - expected_parameters[name].detach()).abs().max()
                    ),
                )
            losses.append(float(loss.detach()))
        with torch.no_grad():
            final = model(**batch)
            train_after = float((final - target).square().mean())
            valid_after = float((model(**validation) - validation_target).square().mean())
        exported = model.export(os.path.join(directory, "export"))
        exported_frozen = frozen_hashes(exported)
        assert {name: exported_frozen[name] for name in frozen} == frozen
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=update.state_dict(),
                step=start_step + steps,
                data=data_identity,
                config=config,
            ),
            os.path.join(directory, "checkpoint.pt"),
        )
    with OnnxModule.load(os.path.join(directory, "export")) as restored, torch.no_grad():
        torch.testing.assert_close(restored(**batch), final, rtol=0, atol=0)
        for name, parameter in restored.parameters_by_name().items():
            expected_parameters[name].copy_(parameter)
        initial = batch["sample"]
        sampled = sample_latents(
            restored, schedule_config, initial, batch["encoder_hidden_states"]
        )
        expected_sample = sample_latents(
            lambda **kw: reference(**kw).sample,
            schedule_config,
            initial,
            batch["encoder_hidden_states"],
        )
        sampling_error = compare_prediction(sampled, expected_sample)
    export = os.path.join(directory, "export")
    for entry in os.scandir(export):
        if entry.name.endswith(".onnx"):
            onnx.checker.check_model(entry.path, full_check=True)
    report = dict(
        model=MODEL_ID if real else "offline-unet",
        revision=REVISION if real else None,
        training_data="external encoded latents"
        if batch_file
        else "fixed synthetic latents and conditioning",
        data=data_identity,
        latent_shape=list(batch["sample"].shape),
        steps=steps,
        resumed_step=start_step,
        train_loss=[train_before, train_after],
        validation_loss=[valid_before, valid_after],
        losses=losses,
        adapter_mapping=mapping,
        maximum_gradient_error=gradient_error,
        maximum_parameter_error=parameter_error,
        parameter_trajectory_diagnostics=parameter_diagnostics,
        forward_trajectory_diagnostics=forward_diagnostics,
        frozen_weights_unchanged=True,
        loaded_prediction_bitwise_equal=True,
        external_scheduler_sampling_aligned=True,
        sampling_error=sampling_error,
        prediction_type=scheduler.config.prediction_type,
        graph_optimizer="ONNXScript",
        ort_graph_optimizer=False,
        module_construction_seconds=construction_seconds,
        module_construction_scope="VJP generation, ONNXScript optimization, serialization and session creation",
        torch_threads=1,
        ort_intra_op_threads=1,
        ort_inter_op_threads=1,
        graph_nodes=len(graph.graph),
        trainable_parameter_bytes=sum(value.nbytes for value in params.values()),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if os.uname().sysname == "Darwin" else 1024),
        versions={
            name: version(name)
            for name in ("diffusers", "torch", "onnxruntime", "mobius-onnx")
        },
    )
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument(
        "--batch", help="Torch tensor file: clean, conditioning, noise, timesteps"
    )
    parser.add_argument("--resume")
    args = parser.parse_args()
    report = run(
        args.output_dir,
        steps=args.steps,
        real=args.pretrained,
        size=args.latent_size,
        batch_file=args.batch,
        resume=args.resume,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("model", "steps", "train_loss", "validation_loss")}
        )
    )


if __name__ == "__main__":
    main()
