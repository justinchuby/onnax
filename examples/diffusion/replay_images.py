# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Verify real-image checkpoint recovery and execute both saved ONNX bundles."""

import argparse
import json
import os

import onnx
import torch
from diffusers import DDPMScheduler
from huggingface_hub import hf_hub_download

from examples.diffusion.objective import DiagnosticSessions
from examples.diffusion.train_images import encoded_data
from examples.diffusion.unet import MODEL_ID, REVISION
from examples.diffusion.unet_train import training_batch
from examples.huggingface.block_runtime import digest
from examples.huggingface.replay import _equal_tree
from onnax.torch import OnnxModule


def _tensor_count(value):
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(_tensor_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_count(v) for v in value)
    return 0


def verify(first, resumed, encoded_directory, *, step=20, provider="mlx", library=None):
    if provider not in ("mlx", "cpu"):
        raise ValueError("Select mlx/cpu")
    torch.set_num_threads(1)
    from examples.lora.train import session as cpu_session

    factory = cpu_session
    if provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if library:
            register_library(library)
        factory = session
    elif library:
        raise ValueError("An MLX library requires provider=mlx")
    states = [
        torch.load(
            os.path.join(root, f"checkpoint-{step}.pt"), weights_only=True, map_location="cpu"
        )
        for root in (first, resumed)
    ]
    keys = ("model", "optimizer", "step", "data", "source_identity")
    for name in keys:
        _equal_tree(states[0][name], states[1][name])
    _, splits, identity = encoded_data(encoded_directory)
    if identity != states[0]["data"]:
        raise ValueError("Encoded recovery data differs from checkpoint")
    objectives = []
    for root in (first, resumed):
        with open(os.path.join(root, "report.json"), encoding="utf-8") as stream:
            objectives.append(json.load(stream).get("objective"))
    _equal_tree(*objectives)
    objective = objectives[0]
    interfaces = []
    models = 0
    predictions = []
    for root in (first, resumed):
        export = os.path.join(root, "export")
        with open(os.path.join(export, "interface.json"), encoding="utf-8") as stream:
            interface = json.load(stream)
        for name, checksum in interface["files"].items():
            if (
                os.path.basename(name) != name
                or digest(os.path.join(export, name)) != checksum
            ):
                raise ValueError("Exported file checksum differs")
            if name.endswith(".onnx"):
                onnx.checker.check_model(os.path.join(export, name), full_check=True)
                models += 1
        if objective:
            prediction_path = os.path.join(export, "prediction.onnx")
            onnx.checker.check_model(prediction_path, full_check=True)
            models += 1
            predictions.append(digest(prediction_path))
        interfaces.append(interface)
    _equal_tree(*interfaces)
    if predictions:
        _equal_tree(*predictions)
    path = hf_hub_download(MODEL_ID, "scheduler/scheduler_config.json", revision=REVISION)
    with open(path, encoding="utf-8") as stream:
        scheduler = DDPMScheduler.from_config(json.load(stream))
    row = splits["train"][0]
    batch, target = training_batch(
        row["clean"], row["conditioning"], row["noise"], row["timesteps"], scheduler
    )
    results = []
    frozen_predictions = []
    frozen_errors = []
    for root in (first, resumed):
        diagnostics = DiagnosticSessions(factory, objective) if objective else None
        with OnnxModule.load(
            os.path.join(root, "export"), session_factory=diagnostics or factory
        ) as model:
            if objective:
                loss = model(**batch, **{objective["target"]: target})
                prediction = torch.from_numpy(diagnostics.prediction)
            else:
                prediction = model(**batch)
                loss = (prediction - target).square().mean()
            loss.backward()
            values = dict(
                prediction=prediction.detach().clone(),
                loss=loss.detach().clone(),
                **{
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.parameters_by_name().items()
                },
            )
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise AssertionError("Nonfinite recovered output")
            results.append(values)
        if objective:
            inference = factory(os.path.join(root, "export", "prediction.onnx"))
            frozen = torch.from_numpy(
                inference.run(None, {name: value.numpy() for name, value in batch.items()})[0]
            )
            torch.testing.assert_close(frozen, values["prediction"], rtol=1e-4, atol=1e-6)
            frozen_predictions.append(frozen.clone())
            frozen_errors.append(float((frozen - values["prediction"]).abs().max()))
            del inference
    _equal_tree(*results)
    if objective:
        _equal_tree(*frozen_predictions)
    return dict(
        step=step,
        state_tensors_byte_identical=sum(_tensor_count(states[0][name]) for name in keys),
        checkpoint_keys_verified=list(keys),
        export_interfaces_identical=True,
        exported_file_checksums_identical=len(interfaces[0]["files"]) + bool(objective),
        eager_gradients=bool(objective),
        frozen_inference_max_absolute_error=max(frozen_errors) if frozen_errors else None,
        full_checker_models=models,
        loaded_tensors_byte_identical=len(results[0]),
        provider=provider,
        cpu_fallback=False,
        ort_graph_optimizer=False,
        latent_shape=list(batch["sample"].shape),
        encoded_data_verified=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--encoded-dir", required=True)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--provider", choices=("mlx", "cpu"), default="mlx")
    parser.add_argument("--mlx-library")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(
        args.first,
        args.resumed,
        args.encoded_dir,
        step=args.step,
        provider=args.provider,
        library=args.mlx_library,
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
