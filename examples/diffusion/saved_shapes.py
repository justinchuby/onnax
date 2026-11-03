# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure runtime shape-vector boundaries for a real UNet vector pullback."""

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

import onnax
from examples.diffusion.benchmark_unet import load_state, statistics
from examples.diffusion.unet import MODEL_ID, REVISION
from examples.diffusion.unet_train import training_batch
from examples.huggingface.block_runtime import MlxMemory, arrays_record, digest
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.residuals import split_vjp
from examples.huggingface.serialization import SharedWeights
from examples.huggingface.shape_residuals import compress
from examples.lora.mlx_runtime import register_library, session
from examples.process_memory import physical_footprint


def measure(
    directory,
    batch_file,
    output,
    *,
    library,
    save_shapes=False,
    fused_pullbacks=False,
    shape_vector_adjoints=False,
    warmup=10,
    samples=100,
):
    """Compare saved-boundary execution with a whole-VJP result at fixed inputs."""
    if shape_vector_adjoints and not fused_pullbacks:
        raise ValueError("Shape-vector adjoints require fused pullbacks")
    if warmup < 0 or samples < 1:
        raise ValueError("Use nonnegative warmup and positive sample counts")
    os.makedirs(output, exist_ok=False)
    torch.set_num_threads(1)
    register_library(library)
    _, interface, parameters = load_state(directory)
    export = os.path.join(directory, "export")
    for name, checksum in interface["files"].items():
        if os.path.basename(name) != name or digest(os.path.join(export, name)) != checksum:
            raise ValueError(f"Source export changed: {name}")
    source = ir.load(os.path.join(export, "model.onnx"))
    encoded = torch.load(batch_file, weights_only=True, map_location="cpu")
    scheduler_path = hf_hub_download(
        MODEL_ID, "scheduler/scheduler_config.json", revision=REVISION
    )
    with open(scheduler_path, encoding="utf-8") as stream:
        schedule = json.load(stream)
    batch, target = training_batch(
        encoded["clean"],
        encoded["conditioning"],
        encoded["noise"],
        encoded["timesteps"],
        DDPMScheduler.from_config(schedule),
    )
    feeds = {**parameters, **{name: value.numpy() for name, value in batch.items()}}
    input_record = arrays_record(feeds)
    joint = onnax.vjp(source, inputs=list(parameters))
    if fused_pullbacks:
        joint = onnax.FuseAttentionVJPPass(recompute=True)(joint).model
        joint = onnax.FuseBroadcastAdjointsPass(shape_inputs=shape_vector_adjoints)(
            joint
        ).model
    derivative_interface = json.loads(joint.graph.metadata_props["onnax.autodiff.interface"])
    (seed,) = derivative_interface["seeds"]
    gradients = derivative_interface["derivative_outputs"]
    forward, backward, residuals = split_vjp(source, joint, optimize_layout=True)
    backward = optimize_training_graph(backward)
    live = {
        value.name
        for value in backward.graph.inputs
        if value.uses() or value in backward.graph.outputs
    }
    residuals = tuple(name for name in residuals if name in live)
    for value in list(backward.graph.inputs):
        if value.name not in live:
            backward.graph.inputs.remove(value)
    original_outputs = [value.name for value in source.graph.outputs]
    selected = set(original_outputs) | set(residuals)
    kept = [value for value in forward.graph.outputs if value.name in selected]
    forward.graph.outputs.clear()
    forward.graph.outputs.extend(kept)
    plan = None
    if save_shapes:
        forward, backward, residuals, plan = compress(
            forward, backward, residuals, outputs=original_outputs
        )
    forward, backward = map(optimize_training_graph, (forward, backward))
    memory = MlxMemory(library)
    with SharedWeights(output) as store:
        paths = {
            name: store.save(model, name)
            for name, model in (
                ("joint", optimize_training_graph(joint)),
                ("forward", forward),
                ("backward", backward),
            )
        }
    for path in paths.values():
        onnx.checker.check_model(path, full_check=True)
    reference = session(paths["joint"])
    prediction = reference.run(
        original_outputs, {**feeds, seed: np.zeros(tuple(target.shape), np.float32)}
    )[0]
    cotangent = (prediction - target.numpy()) * np.float32(2 / target.numel())
    expected = reference.run(gradients, {**feeds, seed: cotangent})
    del reference, joint
    gc.collect()
    before = time.perf_counter()
    fwd, bwd = session(paths["forward"]), session(paths["backward"])
    creation = time.perf_counter() - before
    backward_inputs = {v.name for v in backward.graph.inputs}
    boundary_names = [value.name for value in forward.graph.outputs]

    def execute():
        begin = time.perf_counter()
        values = dict(zip(boundary_names, fwd.run(boundary_names, feeds), strict=True))
        boundary = time.perf_counter()
        supplied = {
            name: value
            for name, value in {**feeds, seed: cotangent, **values}.items()
            if name in backward_inputs
        }
        actual = bwd.run(gradients, supplied)
        finish = time.perf_counter()
        return (
            actual,
            sum(values[name].nbytes for name in residuals),
            (boundary - begin, finish - boundary, finish - begin),
        )

    def check(actual):
        for name, a, b in zip(gradients, actual, expected, strict=True):
            if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all():
                raise AssertionError(f"Invalid gradient interface or values: {name}")
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6, err_msg=name)

    actual, saved_bytes, _ = execute()
    check(actual)
    del actual
    for _ in range(warmup):
        execute()
    memory.reset_peak()
    timings = []
    for _ in range(samples):
        actual, _, timing = execute()
        timings.append(timing)
    check(actual)
    if arrays_record(feeds) != input_record:
        raise AssertionError("Runtime mutated the source inputs")
    result = dict(
        version=1,
        source_identity=interface["identity"],
        save_shapes=save_shapes,
        fused_pullbacks=fused_pullbacks,
        shape_vector_adjoints=shape_vector_adjoints,
        inputs=input_record,
        library_sha256=digest(library),
        graph_sha256={name: digest(path) for name, path in paths.items()},
        versions={
            name: version(name)
            for name in ("numpy", "onnx", "onnx-ir", "onnxruntime", "torch")
        },
        torch_threads=torch.get_num_threads(),
        environment={
            name: os.environ.get(name)
            for name in (
                "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
            )
        },
        residual_count=len(residuals),
        saved_bytes=saved_bytes,
        shape_plan=plan,
        gradients_checked=len(gradients),
        warmup=warmup,
        samples=samples,
        session_creation_seconds=creation,
        timing_seconds={
            name: statistics([row[i] for row in timings])
            for i, name in enumerate(("forward", "backward", "paired"))
        },
        allocator=memory.snapshot(),
        physical_footprint=physical_footprint(),
        scope="Fixed-input split vector VJP; includes host boundary arrays; excludes loss and optimizer. Physical peak includes golden whole-VJP qualification.",
        cpu_fallback=False,
        ort_graph_optimizer=False,
    )
    with open(os.path.join(output, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlx-library", required=True)
    parser.add_argument("--save-shapes", action="store_true")
    parser.add_argument("--fused-pullbacks", action="store_true")
    parser.add_argument("--shape-vector-adjoints", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    result = measure(
        args.training_run,
        args.batch,
        args.output_dir,
        library=args.mlx_library,
        save_shapes=args.save_shapes,
        fused_pullbacks=args.fused_pullbacks,
        shape_vector_adjoints=args.shape_vector_adjoints,
        warmup=args.warmup,
        samples=args.samples,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("saved_bytes", "residual_count", "timing_seconds")}
        )
    )


if __name__ == "__main__":
    main()
