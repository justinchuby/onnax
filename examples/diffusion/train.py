# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train an offline conditional DiT through ONNX VJP and an external AdamW loop."""

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version

import onnx
import onnxruntime as ort
import torch
from onnxscript import optimizer

from examples.diffusion.dit import build, inject
from examples.diffusion.reference import forward
from onnax.torch import OnnxModule


def data(count, seed):
    """External fixed noise schedule: x_t = sqrt(a_t) x_0 + sqrt(1-a_t) noise."""
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randn(count, 4, 8, 8, generator=generator) * 0.3
    noise = torch.randn(count, 4, 8, 8, generator=generator)
    timestep = torch.randint(1, 100, (count,), generator=generator).float()
    alpha = torch.cos(timestep / 100 * (torch.pi / 2)).square()[:, None, None, None]
    return {
        "sample": alpha.sqrt() * clean + (1 - alpha).sqrt() * noise,
        "timestep": timestep,
        "encoder_hidden_states": torch.randn(count, 5, 32, generator=generator),
    }, noise


def frozen_hashes(model):
    return {
        name: hashlib.sha256(value.const_value.numpy().tobytes()).hexdigest()
        for name, value in model.graph.initializers.items()
    }


def run(directory, *, steps=100, resume=None):
    """Compare every forward, gradient and update against independent Torch equations."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("Use at least two training updates")
    os.makedirs(directory, exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    working_tree_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    config, base, weights = build()
    graph, params, mapping = inject(base)
    frozen = frozen_hashes(graph)
    constants = {name: torch.from_numpy(value) for name, value in weights.items()}
    expected_params = {
        name: torch.nn.Parameter(torch.from_numpy(value.copy()))
        for name, value in params.items()
    }
    train, noise = data(16, 20261102)
    validation, validation_noise = data(8, 20261103)
    losses, max_gradient_error, max_parameter_error = [], 0.0, 0.0
    trajectory_forward_diagnostics = []
    trajectory_parameter_diagnostics = []
    data_identity = {
        name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
        for name, value in {**train, "noise": noise}.items()
    }
    start_step = 0
    with OnnxModule(graph, params, input_gradients=[], optimize=optimizer.optimize) as model:
        update = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0, eps=1e-5)
        reference_update = torch.optim.AdamW(
            expected_params.values(), lr=0.01, weight_decay=0, eps=1e-5
        )
        if resume is not None:
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if state["data"] != data_identity:
                raise ValueError("Checkpoint denoising data changed")
            model.load_state_dict(state["model"])
            update.load_state_dict(state["optimizer"])
            with torch.no_grad():
                for name, parameter in model.parameters_by_name().items():
                    expected_params[name].copy_(parameter)
            reference_update.load_state_dict(copy.deepcopy(state["optimizer"]))
            start_step = state["step"]
        with torch.no_grad():
            validation_before = float((model(**validation) - validation_noise).square().mean())
        for step in range(steps):
            update.zero_grad(set_to_none=True)
            reference_update.zero_grad(set_to_none=True)
            actual = model(**train)
            expected = forward(config, constants, expected_params, mapping, **train)
            # Local forward correctness uses identical parameter values. Independent
            # updates retain their own rounding history, checked below through loss,
            # gradients and parameters; record their pointwise output differences.
            shared = {
                name: parameter.detach()
                for name, parameter in model.parameters_by_name().items()
            }
            with torch.no_grad():
                local = forward(config, constants, shared, mapping, **train)
            torch.testing.assert_close(actual, local, rtol=1e-4, atol=1e-6)
            difference = (actual.detach() - expected.detach()).abs()
            trajectory_forward_diagnostics.append(
                {
                    "step": step,
                    "max_absolute_error": float(difference.max()),
                    "rms_error": float(difference.square().mean().sqrt()),
                    "pointwise_tolerance_failures": int(
                        (difference > 1e-6 + 1e-4 * expected.detach().abs()).sum()
                    ),
                }
            )
            loss = (actual - noise).square().mean()
            reference_loss = (expected - noise).square().mean()
            torch.testing.assert_close(loss, reference_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            reference_loss.backward()
            for name, parameter in model.parameters_by_name().items():
                ref = expected_params[name]
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"Missing or nonfinite adapter gradient: {name}")
                torch.testing.assert_close(parameter.grad, ref.grad, rtol=1e-4, atol=1e-6)
                max_gradient_error = max(
                    max_gradient_error, float((parameter.grad - ref.grad).abs().max())
                )
                if (
                    start_step == 0
                    and step == 0
                    and name.endswith(".A")
                    and torch.count_nonzero(parameter.grad)
                ):
                    raise AssertionError(
                        "Zero-initialized B must give zero initial A gradient"
                    )
            if start_step == 0 and step <= 1:
                for name, parameter in model.parameters_by_name().items():
                    if (step == 0 and name.endswith(".B")) or (
                        step == 1 and name.endswith(".A")
                    ):
                        if not torch.count_nonzero(parameter.grad):
                            raise AssertionError(
                                f"Expected an effective adapter gradient: {name}"
                            )
            update.step()
            reference_update.step()
            for name, parameter in model.parameters_by_name().items():
                ref = expected_params[name]
                difference = parameter.detach() - ref.detach()
                error_norm = float(torch.linalg.vector_norm(difference))
                reference_norm = float(torch.linalg.vector_norm(ref.detach()))
                if error_norm > 1e-6 + 1e-4 * reference_norm:
                    raise AssertionError(
                        f"Independent parameter norm drift at update {step}: {name}"
                    )
                trajectory_parameter_diagnostics.append(
                    {
                        "step": step,
                        "parameter": name,
                        "absolute_l2_error": error_norm,
                        "reference_l2_norm": reference_norm,
                        "pointwise_tolerance_failures": int(
                            (difference.abs() > 1e-6 + 1e-4 * ref.detach().abs()).sum()
                        ),
                    }
                )
                max_parameter_error = max(
                    max_parameter_error, float((parameter.detach() - ref.detach()).abs().max())
                )
            losses.append(float(loss.detach()))
        with torch.no_grad():
            prediction = model(**validation)
            validation_after = float((prediction - validation_noise).square().mean())
            train_after = float((model(**train) - noise).square().mean())
            reference_train_after = float(
                (forward(config, constants, expected_params, mapping, **train) - noise)
                .square()
                .mean()
            )
            reference_validation_after = float(
                (
                    forward(config, constants, expected_params, mapping, **validation)
                    - validation_noise
                )
                .square()
                .mean()
            )
            torch.testing.assert_close(
                torch.tensor(train_after),
                torch.tensor(reference_train_after),
                rtol=1e-4,
                atol=1e-6,
            )
            torch.testing.assert_close(
                torch.tensor(validation_after),
                torch.tensor(reference_validation_after),
                rtol=1e-4,
                atol=1e-6,
            )
        bundle = os.path.join(directory, "export")
        bound = model.export(bundle)
        if any(frozen_hashes(bound).get(name) != digest for name, digest in frozen.items()):
            raise AssertionError("Frozen DiT weights changed")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": update.state_dict(),
                "step": start_step + steps,
                "data": data_identity,
            },
            os.path.join(directory, "checkpoint.pt"),
        )
        with OnnxModule.load(bundle) as loaded:
            restored = loaded(**validation)
            if (
                restored.dtype != prediction.dtype
                or restored.shape != prediction.shape
                or restored.detach().numpy().tobytes() != prediction.detach().numpy().tobytes()
            ):
                raise AssertionError("Loaded denoiser output bytes differ")
        for name in ("model.onnx", "execution.onnx", "derivative.onnx", "forward.onnx"):
            onnx.checker.check_model(os.path.join(bundle, name), full_check=True)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        bound_session = ort.InferenceSession(
            os.path.join(bundle, "forward.onnx"), options, providers=["CPUExecutionProvider"]
        )
        bound_prediction = torch.from_numpy(
            bound_session.run(
                None, {name: value.numpy() for name, value in validation.items()}
            )[0]
        )
        torch.testing.assert_close(bound_prediction, prediction, rtol=1e-4, atol=1e-6)
    if not train_after < losses[0]:
        raise AssertionError("Fixed training loss did not decrease")
    report = {
        "source_commit": source_commit,
        "working_tree_dirty": working_tree_dirty,
        "versions": {
            name: version(name)
            for name in (
                "onnax",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnxruntime",
                "torch",
                "mobius-onnx",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": {"torch": torch.get_num_threads(), "ort_intra": 1, "ort_inter": 1},
        "config": dataclasses.asdict(config),
        "steps": steps,
        "start_step": start_step,
        "total_updates": start_step + steps,
        "data_identity": data_identity,
        "reference_final_training_loss": reference_train_after,
        "reference_final_validation_loss": reference_validation_after,
        "task": "epsilon_prediction",
        "device": "cpu",
        "precision": "fp32",
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 0.01,
            "epsilon": 1e-5,
            "weight_decay": 0,
        },
        "training_examples": 16,
        "validation_examples": 8,
        "training_seed": 20261102,
        "validation_seed": 20261103,
        "loss_reduction": "mean over batch/channel/height/width",
        "training_loss": [losses[0], train_after],
        "validation_loss": [validation_before, validation_after],
        "losses": losses,
        "independent_trajectory_forward_diagnostics": trajectory_forward_diagnostics,
        "shared_parameter_forward_aligned": True,
        "adapter_mapping": mapping,
        "frozen_sha256": frozen,
        "frozen_weights_unchanged": True,
        "export_load_bitwise_equal": True,
        "bound_forward_loaded_and_executed": True,
        "adapter_gradients_aligned": True,
        "independent_updates_norm_aligned": True,
        "parameter_trajectory_policy": "L2 error <= 1e-6 + 1e-4 * reference parameter L2 norm; original pointwise failures retained",
        "independent_trajectory_parameter_diagnostics": trajectory_parameter_diagnostics,
        "max_gradient_absolute_error": max_gradient_error,
        "max_parameter_absolute_error": max_parameter_error,
        "ort_graph_optimization": "disabled",
        "graph_optimizer": "ONNXScript",
        "elapsed_seconds": time.perf_counter() - start,
        "measurement_scope": "Whole CPU qualification including independent references, update comparisons, ONNX export and reload",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--resume", help="External checkpoint.pt from a previous run")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, steps=args.steps, resume=args.resume)))


if __name__ == "__main__":
    main()
