# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify the public saved-activation bridge against a whole UNet VJP."""

import argparse
import gc
import json
import os
import time
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
import torch
from diffusers import DDPMScheduler
from huggingface_hub import hf_hub_download

from examples.diffusion.benchmark_unet import load_state
from examples.diffusion.unet import MODEL_ID, REVISION
from examples.diffusion.unet_train import training_batch
from examples.huggingface.block_runtime import arrays_record, digest
from examples.huggingface.optimization import optimize_training_graph
from examples.lora.mlx_runtime import register_library, session
from examples.process_memory import physical_footprint
from onnax.torch import OnnxModule


def qualify(source_dir, batch_file, golden_path, *, library, output):
    """Compare predictions and every adapter gradient at fixed real-image inputs."""
    torch.set_num_threads(1)
    register_library(library)
    _, interface, parameters = load_state(source_dir)
    source_path = source_dir + "/export/model.onnx"
    source_checksum = digest(source_path)
    encoded = torch.load(batch_file, weights_only=True, map_location="cpu")
    with open(
        hf_hub_download(MODEL_ID, "scheduler/scheduler_config.json", revision=REVISION)
    ) as stream:
        scheduler = DDPMScheduler.from_config(json.load(stream))
    batch, target = training_batch(
        encoded["clean"],
        encoded["conditioning"],
        encoded["noise"],
        encoded["timesteps"],
        scheduler,
    )
    feeds = {**parameters, **{n: v.numpy() for n, v in batch.items()}}
    input_record = arrays_record(feeds)
    golden_model = ir.load(golden_path)
    record = json.loads(golden_model.graph.metadata_props["onnax.autodiff.interface"])
    (seed,) = record["seeds"]
    names = record["derivative_outputs"]
    if record["selection"]["inputs"] != list(parameters):
        raise ValueError("Whole-VJP parameter mapping differs from the adapter interface")
    output_name = golden_model.graph.outputs[0].name
    golden = session(golden_path)
    prediction = golden.run(
        [output_name], {**feeds, seed: np.zeros(tuple(target.shape), np.float32)}
    )[0]
    cotangent = (prediction - target.numpy()) * np.float32(2 / target.numel())
    expected = golden.run(names, {**feeds, seed: cotangent})
    del golden, golden_model
    gc.collect()
    print("Golden whole-VJP complete", flush=True)
    retained = {}

    def factory(path):
        runtime = session(path)

        class Session:
            def run(self, names, feeds):
                arrays = runtime.run(names, feeds)
                if path.endswith("execution.onnx"):
                    retained.update((n, a.nbytes) for n, a in zip(names, arrays, strict=True))
                return arrays

        return Session()

    start = time.perf_counter()
    with OnnxModule(
        source_path,
        parameters,
        input_gradients=[],
        save_activations=True,
        optimize=optimize_training_graph,
        session_factory=factory,
    ) as model:
        creation = time.perf_counter() - start
        print("Public saved module created", flush=True)
        for stem in ("model", "execution", "derivative"):
            onnx.checker.check_model(
                os.path.join(model._directory.name, stem + ".onnx"), full_check=True
            )
        actual = model(**batch)
        np.testing.assert_allclose(actual.detach().numpy(), prediction, rtol=1e-4, atol=1e-6)
        loss = (actual - target).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite public objective")
        loss.backward()
        for (name, parameter), wanted in zip(
            model.parameters_by_name().items(), expected, strict=True
        ):
            if not np.isfinite(parameter.grad.numpy()).all():
                raise AssertionError(f"Nonfinite gradient: {name}")
            np.testing.assert_allclose(
                parameter.grad.numpy(), wanted, rtol=1e-4, atol=1e-6, err_msg=name
            )
        report = dict(
            source_identity=interface["identity"],
            library_sha256=digest(library),
            parameters_checked=len(expected),
            prediction_aligned=True,
            gradient_rtol=1e-4,
            gradient_atol=1e-6,
            loss=loss.item(),
            saved_bytes=sum(retained[n] for n in model._saved_names),
            residual_count=len(model._saved_names),
            session_and_graph_creation_seconds=creation,
            physical_footprint=physical_footprint(),
            input_arrays=input_record,
            driver_sha256=digest(__file__),
            full_checker_graphs=3,
            cpu_fallback=False,
            ort_graph_optimizer=False,
        )
    if arrays_record(feeds) != input_record:
        raise AssertionError("Runtime mutated the source inputs or adapters")
    if digest(source_path) != source_checksum:
        raise AssertionError("Source model mutated")
    report["versions"] = {
        name: version(name)
        for name in ("numpy", "onnx", "onnx-ir", "onnxscript", "onnxruntime", "torch")
    }
    report["golden_graph_sha256"] = digest(golden_path)
    report["scope"] = (
        "One public OnnxModule FP32 real UNet forward and external MSE backward, compared with a whole-VJP at identical inputs and adapters; process memory includes setup and both execution paths."
    )
    with open(output, "w") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["parameters_checked", "saved_bytes", "residual_count", "loss"]
            }
        ),
        flush=True,
    )


def main():
    """Run a native MLX qualification using existing pinned workload artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--batch-file", required=True)
    parser.add_argument("--golden-path", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    qualify(**vars(args))


if __name__ == "__main__":
    main()
