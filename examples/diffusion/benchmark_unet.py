# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure a saved UNet training step on matched real image latents using MLX/MPS."""

import argparse
import gc
import json
import os
import platform
import resource
import time
from importlib.metadata import version

import numpy as np
import torch
from diffusers import DDPMScheduler
from huggingface_hub import hf_hub_download

from examples.diffusion.objective import DiagnosticSessions, read_objective
from examples.diffusion.unet import MODEL_ID, REVISION, pretrained
from examples.diffusion.unet_reference import make_reference
from examples.diffusion.unet_train import training_batch
from examples.huggingface.block_runtime import MlxMemory, arrays_record, digest
from examples.process_memory import physical_footprint
from examples.torch_deploy import install_header, read_header
from onnax.torch import OnnxModule


def statistics(values):
    return dict(
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        samples=len(values),
    )


def persist_progress(
    output, *, warmup, samples, completed_warmup, records, latest, allocator=None
):
    """Preserve completed samples if an external guard terminates the worker."""
    path = output + ".progress.json"
    record = dict(
        version=1,
        requested_warmup=warmup,
        completed_warmup=completed_warmup,
        requested_samples=samples,
        completed_samples=len(records),
        sample_seconds=records,
        latest_step_seconds=latest,
        allocator=allocator,
        status="sampling_complete"
        if len(records) == samples
        else "sampling"
        if records
        else "warmup",
    )
    with open(path + ".tmp", "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    os.replace(path + ".tmp", path)


def load_state(directory):
    """Load the stable adapter mapping and verify the saved parameter payload."""
    with open(os.path.join(directory, "report.json"), encoding="utf-8") as stream:
        report = json.load(stream)
    if (report["model"], report["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Expected the pinned pretrained UNet training run")
    export = os.path.join(directory, "export")
    with open(os.path.join(export, "interface.json"), encoding="utf-8") as stream:
        interface = json.load(stream)
    path = os.path.join(export, "parameters.npz")
    if digest(path) != interface["files"]["parameters.npz"]:
        raise ValueError("Saved adapter payload checksum mismatch")
    names = interface["specification"]["parameters"]
    with np.load(path) as arrays:
        params = {name: arrays[f"arr_{i}"].copy() for i, name in enumerate(names)}
    return report, interface, params


def measure(
    directory,
    batch_file,
    *,
    provider,
    output,
    library=None,
    warmup=10,
    samples=100,
    component_samples=10,
    checkpoint=False,
    derivative_header=None,
    objective_export=None,
):
    """Time natural full steps; collect synchronized phase profiles separately."""
    if provider not in ("mlx", "mps") or samples < 1 or warmup < 0 or component_samples < 0:
        raise ValueError(
            "Select mlx/mps, positive samples and nonnegative warmup/component counts"
        )
    if provider != "mps" and checkpoint:
        raise ValueError("Gradient checkpointing selects the MPS reference route")
    if objective_export and (provider != "mlx" or derivative_header):
        raise ValueError(
            "Objective exports select the MLX route without a derivative override"
        )
    if provider != "mlx" and derivative_header:
        raise ValueError("Derivative deployment headers require provider=mlx")
    if provider == "mps" and library:
        raise ValueError("An MLX library requires provider=mlx")
    if provider == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS measurements require CPU fallback disabled")
    output = os.fspath(output)
    if any(os.path.exists(output + suffix) for suffix in ("", ".npz", ".progress.json")):
        raise FileExistsError("Use a new benchmark output path")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.set_num_threads(1)
    source, interface, initial = load_state(directory)
    deployment = read_header(derivative_header, interface) if derivative_header else None
    objective = None
    if objective_export:
        objective = read_objective(objective_export, interface)
    encoded = torch.load(batch_file, weights_only=True, map_location="cpu")
    path = hf_hub_download(MODEL_ID, "scheduler/scheduler_config.json", revision=REVISION)
    with open(path, encoding="utf-8") as stream:
        schedule = json.load(stream)
    batch, target = training_batch(
        encoded["clean"],
        encoded["conditioning"],
        encoded["noise"],
        encoded["timesteps"],
        DDPMScheduler.from_config(schedule),
    )
    setup = time.perf_counter()
    runtime_calls = []
    native_memory = None
    if provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if library:
            register_library(library)
            native_memory = MlxMemory(library)

        diagnostics = DiagnosticSessions(session, objective) if objective else None

        class TimedSession:
            def __init__(self, path):
                self.name = os.path.basename(path)
                if deployment is not None:
                    path = install_header(path, deployment)
                self.session = diagnostics(path) if diagnostics else session(path)

            def run(self, names, feeds):
                start = time.perf_counter()
                result = self.session.run(names, feeds)
                runtime_calls.append((self.name, time.perf_counter() - start))
                return result

        model = OnnxModule.load(
            objective_export or os.path.join(directory, "export"), session_factory=TimedSession
        )
        if objective is not None and (
            not model._eager_gradients or model._identity != objective["objective_identity"]
        ):
            model.close()
            raise ValueError("Objective execution interface differs from its manifest")
        params = model.parameters_by_name()
        if arrays_record(model.arrays()) != arrays_record(initial):
            model.close()
            raise ValueError("Objective adapter values differ from the benchmark source")
        predict = model

        def synchronize():
            return None
    else:
        config, weights, _ = pretrained()
        model, params = make_reference(
            config, weights, initial, source["adapter_mapping"], device="mps"
        )
        del weights
        if checkpoint:
            model.enable_gradient_checkpointing()

        def predict(**values):
            return model(**values).sample

        batch = {name: value.to("mps") for name, value in batch.items()}
        target = target.to("mps")
        synchronize = torch.mps.synchronize
    update = torch.optim.AdamW(params.values(), lr=0.0, eps=1e-5, weight_decay=0)
    gc.collect()
    synchronize()
    construction = time.perf_counter() - setup

    def step(*, phases=False):
        start = time.perf_counter()
        update.zero_grad(set_to_none=True)
        if objective is not None:
            loss = predict(**batch, **{objective["target"]: target})
            prediction = torch.from_numpy(diagnostics.prediction)
        else:
            prediction = predict(**batch)
        if phases:
            synchronize()
        forward = time.perf_counter()
        if objective is None:
            loss = (prediction - target).square().mean()
        loss.backward()
        if phases:
            synchronize()
        backward = time.perf_counter()
        update.step()
        synchronize()
        finish = time.perf_counter()
        return (
            loss.detach(),
            prediction.detach(),
            dict(
                total=finish - start,
                forward=forward - start,
                loss_backward=backward - forward,
                optimizer=finish - backward,
            ),
        )

    records = []
    calls = []
    capture = None
    try:
        for index in range(warmup + samples):
            runtime_calls.clear()
            if index == warmup and native_memory:
                native_memory.reset_peak()
            loss, prediction, timing = step()
            if index >= warmup:
                records.append(timing["total"])
                calls.extend(runtime_calls)
            if index == warmup:
                capture = {
                    "prediction": prediction.cpu().numpy().copy(),
                    "loss": loss.cpu().numpy().copy(),
                    **{
                        name: parameter.grad.detach().cpu().numpy().copy()
                        for name, parameter in params.items()
                    },
                }
            if not torch.isfinite(loss).item():
                raise AssertionError("Nonfinite measured training loss")
            if index == warmup:
                np.savez(output + ".npz", **capture)
            persist_progress(
                output,
                warmup=warmup,
                samples=samples,
                completed_warmup=min(index + 1, warmup),
                records=records,
                latest=timing["total"],
                allocator=native_memory.snapshot() if native_memory else None,
            )
            if index < 3 or (index + 1) % 10 == 0:
                print(
                    json.dumps(
                        dict(
                            completed_warmup=min(index + 1, warmup),
                            completed_samples=len(records),
                            latest_step_seconds=timing["total"],
                        )
                    ),
                    flush=True,
                )
        phase_records = {name: [] for name in ("forward", "loss_backward", "optimizer")}
        for _ in range(component_samples):
            _, _, timing = step(phases=True)
            for name in phase_records:
                phase_records[name].append(timing[name])
        if any(
            not np.array_equal(parameter.detach().cpu().numpy(), initial[name])
            for name, parameter in params.items()
        ):
            raise AssertionError("The zero-learning-rate measurement changed adapter values")
        arrays_recorded = arrays_record(
            {name: value.cpu().numpy() for name, value in encoded.items()}
        )
        memory = (
            native_memory.snapshot()
            if native_memory
            else dict(
                current_allocated_bytes=torch.mps.current_allocated_memory(),
                driver_allocated_bytes=torch.mps.driver_allocated_memory(),
            )
        )
        report = dict(
            provider=provider,
            deployment=deployment.manifest if deployment else None,
            objective=objective,
            phase_labels=(
                {
                    "forward": "joint loss/gradient execution",
                    "loss_backward": "autograd handoff",
                    "optimizer": "external update",
                }
                if objective
                else {
                    "forward": "forward execution",
                    "loss_backward": "loss and VJP",
                    "optimizer": "external update",
                }
            ),
            model=MODEL_ID,
            revision=REVISION,
            export_identity=interface["identity"],
            adapter_state=arrays_record(initial),
            data=arrays_recorded,
            dtype="float32",
            latent_shape=list(encoded["clean"].shape),
            conditioning_shape=list(encoded["conditioning"].shape),
            warmup=warmup,
            samples=samples,
            full_step_seconds=statistics(records),
            images_per_second=encoded["clean"].shape[0] / float(np.median(records)),
            phase_seconds={
                name: statistics(values) for name, values in phase_records.items() if values
            },
            phase_scope="Separate phase-profile steps synchronize after forward, backward and optimizer; full steps synchronize only at their final boundary",
            runtime_seconds={
                name: statistics([elapsed for filename, elapsed in calls if filename == name])
                for name in {filename for filename, _ in calls}
            },
            runtime_scope="ORT run includes device transfers and host output materialization; forward wrapper also snapshots CPU inputs",
            environment={
                name: os.environ.get(name)
                for name in (
                    "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                    "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                    "ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES",
                    "ONNXRUNTIME_EP_MLX_GPU_SHAPE_OPS",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "PYTORCH_MPS_FAST_MATH",
                )
            },
            gradient_checkpointing=checkpoint
            if provider == "mps"
            else "whole graph recomputation in VJP",
            learning_rate=0.0,
            optimizer="torch.optim.AdamW",
            optimizer_epsilon=1e-5,
            adapter_state_scope="Fixed adapter values; Adam moments and step advance normally",
            construction_seconds=construction,
            allocator=memory,
            physical_footprint=physical_footprint(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if os.uname().sysname == "Darwin" else 1024),
            memory_scope="Isolated whole process, including model construction, warmup, measurement and capture",
            allocator_scope="MLX peak resets after warmup; MPS counters show current allocator and driver storage after measurement",
            ort_graph_optimizer=False,
            cpu_fallback=False,
            torch_threads=1,
            platform=platform.platform(),
            machine=platform.machine(),
            library_sha256=digest(library) if library else None,
            versions={
                name: version(name)
                for name in ("torch", "onnxruntime", "diffusers", "mobius-onnx")
            },
        )
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        np.savez(output + ".npz", **capture)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        return report
    finally:
        if provider == "mlx":
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--provider", choices=("mlx", "mps"), required=True)
    parser.add_argument("--mlx-library")
    parser.add_argument("--derivative-header")
    parser.add_argument("--objective-export")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--component-samples", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = measure(
        args.training_run,
        args.batch,
        provider=args.provider,
        output=args.output,
        library=args.mlx_library,
        derivative_header=args.derivative_header,
        objective_export=args.objective_export,
        warmup=args.warmup,
        samples=args.samples,
        component_samples=args.component_samples,
        checkpoint=args.checkpoint,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("provider", "full_step_seconds", "images_per_second")
            }
        )
    )


if __name__ == "__main__":
    main()
