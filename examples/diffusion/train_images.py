# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train a saved ONNX UNet on cached image-caption latents with external AdamW."""

import argparse
import copy
import json
import math
import os
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
import torch
from diffusers import DDPMScheduler

from examples.diffusion.benchmark_unet import load_state
from examples.diffusion.objective import DiagnosticSessions, export_prediction, read_objective
from examples.diffusion.train import frozen_hashes
from examples.diffusion.unet import pretrained
from examples.diffusion.unet_reference import make_reference
from examples.diffusion.unet_train import compare_prediction, training_batch
from examples.huggingface.block_runtime import arrays_record
from onnax.torch import OnnxModule


def encoded_data(directory):
    """Verify encoded tensor shape, dtype and content against its source manifest."""
    with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != 1:
        raise ValueError("Expected a version 1 encoded-image manifest")
    splits = {"train": [], "validation": []}
    identity = {}
    for row in manifest["examples"]:
        name = row["output"]
        if os.path.basename(name) != name or row["split"] not in splits:
            raise ValueError("Invalid encoded example path or split")
        values = torch.load(
            os.path.join(directory, name), weights_only=True, map_location="cpu"
        )
        record = arrays_record({key: value.numpy() for key, value in values.items()})
        if record != row["tensors"]:
            raise ValueError(f"Encoded image tensors changed: {name}")
        splits[row["split"]].append(values)
        identity[name] = record
    if not all(splits.values()):
        raise ValueError("Provide both training and validation images")
    return manifest, splits, identity


def run(
    directory,
    encoded_directory,
    output,
    *,
    steps=20,
    library=None,
    resume=None,
    checkpoint_every=10,
    objective_export=None,
):
    """Accumulate image gradients and compare every update with Diffusers on MPS."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS reference validation requires CPU fallback disabled")
    if steps < 1 or checkpoint_every < 1:
        raise ValueError("Use positive total steps and checkpoint interval")
    from examples.lora.mlx_runtime import register_library, session

    if library:
        register_library(library)
    source, interface, initial = load_state(directory)
    objective = read_objective(objective_export, interface) if objective_export else None
    diagnostics = DiagnosticSessions(session, objective) if objective else None
    factory = diagnostics or session

    def execute(module, batch, target):
        if objective:
            loss = module(**batch, **{objective["target"]: target})
            return loss, torch.from_numpy(diagnostics.prediction)
        prediction = module(**batch)
        return (prediction - target).square().mean(), prediction

    manifest, encoded, identity = encoded_data(encoded_directory)
    os.makedirs(output, exist_ok=False)
    torch.set_num_threads(1)
    config, weights, schedule = pretrained()
    scheduler = DDPMScheduler.from_config(schedule)
    batches = {
        name: [
            training_batch(
                row["clean"], row["conditioning"], row["noise"], row["timesteps"], scheduler
            )
            for row in rows
        ]
        for name, rows in encoded.items()
    }
    frozen = frozen_hashes(ir.load(os.path.join(directory, "export", "model.onnx")))
    with OnnxModule.load(
        objective_export or os.path.join(directory, "export"), session_factory=factory
    ) as model:
        if objective and (
            not model._eager_gradients or model._identity != objective["objective_identity"]
        ):
            raise ValueError("Objective execution interface differs from its manifest")
        if arrays_record(model.arrays()) != arrays_record(initial):
            raise ValueError("Objective adapter values differ from the source")
        # Begin image adaptation from a zero-effect adapter at the pretrained base.
        with torch.no_grad():
            for index, row in enumerate(source["adapter_mapping"]):
                a, b = (
                    model.parameters_by_name()[row["A"]],
                    model.parameters_by_name()[row["B"]],
                )
                rng = np.random.default_rng(20261102 + 2 * index)
                a.copy_(
                    torch.from_numpy(rng.normal(0, 0.02, tuple(a.shape)).astype(np.float32))
                )
                b.zero_()
        update = torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, weight_decay=0)
        start = 0
        if resume:
            state = torch.load(resume, weights_only=True, map_location="cpu")
            if state["data"] != identity or state["source_identity"] != interface["identity"]:
                raise ValueError("Checkpoint source or encoded image data changed")
            model.load_state_dict(state["model"])
            update.load_state_dict(state["optimizer"])
            start = state["step"]
        if steps <= start:
            raise ValueError("Total steps must exceed the resumed update")
        reference, parameters = make_reference(
            config, weights, model.arrays(), source["adapter_mapping"], device="mps"
        )
        del weights
        control = torch.optim.AdamW(parameters.values(), lr=0.003, eps=1e-5, weight_decay=0)
        if resume:
            control.load_state_dict(copy.deepcopy(state["optimizer"]))
        controls = {
            name: [
                ({key: value.to("mps") for key, value in batch.items()}, target.to("mps"))
                for batch, target in rows
            ]
            for name, rows in batches.items()
        }

        def evaluate(split):
            with torch.no_grad():
                return sum(
                    float(execute(model, batch, target)[0]) for batch, target in batches[split]
                ) / len(batches[split])

        before = {name: evaluate(name) for name in batches}
        history = []
        for step in range(start, steps):
            update.zero_grad(set_to_none=True)
            control.zero_grad(set_to_none=True)
            losses = []
            predictions = []
            for (batch, target), (reference_batch, reference_target) in zip(
                batches["train"], controls["train"], strict=True
            ):
                objective_loss, actual = execute(model, batch, target)
                expected = reference(**reference_batch).sample
                predictions.append(compare_prediction(actual, expected.cpu()))
                loss = objective_loss / len(batches["train"])
                expected_loss = (expected - reference_target).square().mean() / len(
                    batches["train"]
                )
                torch.testing.assert_close(loss, expected_loss.cpu(), rtol=1e-4, atol=1e-6)
                loss.backward()
                expected_loss.backward()
                losses.append(float(loss.detach()))
            maximum_gradient_error = 0.0
            for name, parameter in model.parameters_by_name().items():
                expected = parameters[name].grad.cpu()
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)
                maximum_gradient_error = max(
                    maximum_gradient_error, float((parameter.grad - expected).abs().max())
                )
            update.step()
            control.step()
            parameter_error = 0.0
            for name, parameter in model.parameters_by_name().items():
                expected = parameters[name].detach().cpu()
                error = float(torch.linalg.vector_norm(parameter.detach() - expected))
                norm = float(torch.linalg.vector_norm(expected))
                if not math.isfinite(error) or error > 1e-6 + 1e-4 * norm:
                    raise AssertionError(f"Independent parameter drift: {name}")
                parameter_error = max(parameter_error, error / max(norm, 1e-30))
            history.append(
                dict(
                    step=step + 1,
                    loss=sum(losses),
                    maximum_gradient_error=maximum_gradient_error,
                    maximum_relative_parameter_l2_error=parameter_error,
                    predictions=predictions,
                )
            )
            if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
                torch.save(
                    dict(
                        step=step + 1,
                        model=model.state_dict(),
                        optimizer=update.state_dict(),
                        data=identity,
                        source_identity=interface["identity"],
                    ),
                    os.path.join(output, f"checkpoint-{step + 1}.pt"),
                )
        after = {name: evaluate(name) for name in batches}
        with torch.no_grad():
            final = {
                name: tuple(value.detach().clone() for value in execute(model, batch, target))
                for name, rows in batches.items()
                for batch, target in rows[-1:]
            }
        bound = model.export(os.path.join(output, "export"))
        if objective:
            export_prediction(
                bound,
                os.path.join(output, "export", "prediction.onnx"),
                objective["prediction"],
                execution=ir.load(os.path.join(output, "export", "execution.onnx")),
            )
        hashes = frozen_hashes(bound)
        if {name: hashes[name] for name in frozen} != frozen:
            raise AssertionError("Frozen UNet weights changed")
    with (
        OnnxModule.load(os.path.join(output, "export"), session_factory=factory) as restored,
        torch.no_grad(),
    ):
        for name, rows in batches.items():
            for actual, expected in zip(
                execute(restored, *rows[-1]), final[name], strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row in os.scandir(os.path.join(output, "export")):
        if row.name.endswith(".onnx"):
            onnx.checker.check_model(row.path, full_check=True)
    report = dict(
        model=source["model"],
        revision=source["revision"],
        source_identity=interface["identity"],
        objective=objective,
        adapter_mapping=source["adapter_mapping"],
        steps=steps,
        resumed_step=start,
        training_images=len(batches["train"]),
        validation_images=len(batches["validation"]),
        gradient_accumulation=len(batches["train"]),
        microbatch=1,
        learning_rate=update.param_groups[0]["lr"],
        optimizer_epsilon=update.param_groups[0]["eps"],
        provider="MLXExecutionProvider",
        reference="Diffusers MPS",
        cpu_fallback=False,
        ort_graph_optimizer=False,
        versions={
            name: version(name)
            for name in ("torch", "diffusers", "onnxruntime", "mobius-onnx")
        },
        adapter_initialization="seeded A, zero B",
        encoded_manifest=manifest,
        data=identity,
        loss_before=before,
        loss_after=after,
        history=history,
        frozen_weights_unchanged=True,
        loaded_predictions_bitwise_equal=True,
    )
    with open(os.path.join(output, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--encoded-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlx-library")
    parser.add_argument("--objective-export")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Total completed updates, including resumed steps",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume")
    args = parser.parse_args()
    report = run(
        args.training_run,
        args.encoded_dir,
        args.output_dir,
        steps=args.steps,
        library=args.mlx_library,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        objective_export=args.objective_export,
    )
    print(json.dumps({key: report[key] for key in ("steps", "loss_before", "loss_after")}))


if __name__ == "__main__":
    main()
