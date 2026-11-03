# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Replay saved UNet gradient diagnostics at an identical adapter parameter point."""

import argparse
import json
import os

import torch
from diffusers import DDPMScheduler

from examples.diffusion.benchmark_unet import load_state
from examples.diffusion.objective import DiagnosticSessions, read_objective
from examples.diffusion.train_images import encoded_data
from examples.diffusion.unet import pretrained
from examples.diffusion.unet_reference import make_reference
from examples.diffusion.unet_train import training_batch
from examples.lora.mlx_runtime import register_library, session
from onnax.torch import OnnxModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--encoded-dir", required=True)
    parser.add_argument("--objective-export", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--mlx-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameter-point", choices=["actual", "reference"], default="actual")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("Replay requires MPS CPU fallback disabled")
    torch.set_num_threads(1)
    root = args.diagnostics_dir
    saved = torch.load(root + "/gradient-mismatch.pt", weights_only=True)
    source, interface, _ = load_state(args.training_run)
    objective = read_objective(args.objective_export, interface)
    register_library(args.mlx_library)
    config, weights, schedule = pretrained()
    scheduler = DDPMScheduler.from_config(schedule)
    _, encoded, identity = encoded_data(args.encoded_dir)
    with open(os.path.join(root, "gradient-mismatch.json"), encoding="utf-8") as stream:
        diagnostic = json.load(stream)
    for key, expected in (
        ("source_identity", interface["identity"]),
        ("objective_identity", objective["objective_identity"]),
        ("data", identity),
    ):
        if key in diagnostic and diagnostic[key] != expected:
            raise ValueError(f"Saved gradient diagnostic differs: {key}")
    batches = [
        training_batch(x["clean"], x["conditioning"], x["noise"], x["timesteps"], scheduler)
        for x in encoded["train"]
    ]
    diag = DiagnosticSessions(session, objective)
    with OnnxModule.load(args.objective_export, session_factory=diag) as model:
        with torch.no_grad():
            for name, p in model.parameters_by_name().items():
                p.copy_(saved[name + ":" + args.parameter_point + "_parameter"])
        reference, parameters = make_reference(
            config,
            weights,
            model.arrays(),
            source["adapter_mapping"],
            device="mps",
            precision=objective.get("projection_precision", "fp32"),
        )
        del weights
        losses = []
        for batch, target in batches:
            loss = model(**batch, **{objective["target"]: target})
            expected = reference(**{k: v.to("mps") for k, v in batch.items()}).sample
            teacher = (expected - target.to("mps")).square().mean()
            (loss / len(batches)).backward()
            (teacher / len(batches)).backward()
            losses.append({"mlx": float(loss.detach()), "mps": float(teacher.detach())})
        rows = {}
        error = norm = dot = actual_norm = 0.0
        for name, p in model.parameters_by_name().items():
            a = p.grad.double()
            b = parameters[name].grad.cpu().double()
            e = float((a - b).square().sum())
            n = float(b.square().sum())
            error += e
            norm += n
            dot += float((a * b).sum())
            actual_norm += float(a.square().sum())
            rows[name] = {
                "relative_l2": (e / n) ** 0.5 if n else None,
                "difference_norm": e**0.5,
                "native_replay_exact": torch.equal(p.grad, saved[name + ":actual_gradient"]),
            }
        result = {
            "losses": losses,
            "gradient_relative_l2": (error / norm) ** 0.5,
            "gradient_cosine": dot / (norm * actual_norm) ** 0.5,
            "parameters": rows,
            "same_parameter_point": True,
            "parameter_point": args.parameter_point,
            "objective_identity": objective["objective_identity"],
            "source_identity": interface["identity"],
            "data": identity,
            "step": diagnostic["step"],
        }
        with open(args.output, "x") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != "parameters"}), flush=True)


if __name__ == "__main__":
    main()
