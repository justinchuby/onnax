# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Replay saved or selectively recomputed block VJPs against captured gradients."""

import argparse
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir

import onnax
from examples.huggingface.block_runtime import (
    MlxMemory,
    arrays_record,
    check_outputs,
    digest,
    load_case,
)
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.recomputation import describe_replay
from examples.huggingface.residuals import split_vjp
from examples.huggingface.serialization import save_graph
from examples.process_memory import physical_footprint


def measure(
    directory,
    captures,
    name,
    *,
    recompute=(),
    checkpoint=False,
    save_shapes=False,
    attention_vjp=False,
    recompute_attention=False,
    factory,
    memory=None,
    warmup=10,
    samples=100,
):
    """Time paired forward/backward calls, retaining residual arrays between them."""
    if warmup < 0 or samples < 1:
        raise ValueError("Expected warmup >= 0 and samples >= 1")
    attention_vjp = attention_vjp or recompute_attention
    recompute = tuple(recompute)
    if checkpoint and (recompute or save_shapes):
        raise ValueError(
            "Whole checkpoint replay and saved-value edits are mutually exclusive"
        )
    if attention_vjp and not checkpoint:
        raise ValueError("Attention deployment qualification requires whole checkpoint replay")
    record, block, feeds, expected = load_case(directory, captures, name)
    row = record["blocks"][block["index"]]
    source = ir.load(os.path.join(directory, f"block_{block['index']}_forward.onnx"))
    if checkpoint:
        forward = source.clone()
        backward = ir.load(os.path.join(directory, block["model"]))
        if attention_vjp:
            from examples.huggingface.attention_vjp import fuse_attention_vjp

            backward = onnax.vjp(
                source, inputs=row["active_inputs"], outputs=row["active_outputs"]
            )
            backward = fuse_attention_vjp(backward, recompute=recompute_attention)
            calls = [
                node.op_type
                for node in backward.graph
                if node.domain == "pkg.onnax"
                and node.op_type.startswith(("AttentionVJP_v1_", "AttentionVJP_v2_"))
            ]
            if not calls:
                raise ValueError("No eligible attention pullback in the selected block")
            values = {value.name: value for value in backward.graph.outputs}
            backward.graph.outputs.clear()
            backward.graph.outputs.extend(values[key] for key in row["gradients"])
            backward = optimize_training_graph(backward)
        residuals = ()
        plan = {
            "version": 1,
            "strategy": "whole_checkpoint",
            "recomputed_values": [],
            "retained_values": [],
            "replay_inputs": list(row["inputs"]),
            "nodes": [],
            "invocations_per_backward": 1,
            "region": "Original exported checkpoint recomputation/VJP",
        }
    else:
        joint = onnax.vjp(source, inputs=row["active_inputs"], outputs=row["active_outputs"])
        _, _, baseline_names = split_vjp(source, joint, optimize_layout=True)
        values = ir.convenience.create_value_mapping(source.graph)
        plan = describe_replay(
            list(source.graph), [values[key] for key in baseline_names], tuple(recompute)
        )
        forward, backward, residuals = split_vjp(
            source, joint, optimize_layout=True, recompute=recompute
        )
        backward = optimize_training_graph(backward)
        plan["strategy"] = "selective" if recompute else "saved"
    plan["attention_vjp_calls"] = calls if attention_vjp else []
    plan["recomputed_attention"] = recompute_attention
    live_inputs = {
        value.name
        for value in backward.graph.inputs
        if value.uses() or value in backward.graph.outputs
    }
    residuals = tuple(key for key in residuals if key in live_inputs)
    for value in list(backward.graph.inputs):
        if value.name not in live_inputs:
            backward.graph.inputs.remove(value)
    original_outputs = {value.name for value in source.graph.outputs}
    retained_outputs = original_outputs | set(residuals)
    outputs = [value for value in forward.graph.outputs if value.name in retained_outputs]
    forward.graph.outputs.clear()
    forward.graph.outputs.extend(outputs)
    if save_shapes:
        from examples.huggingface.shape_residuals import compress

        forward, backward, residuals, shape_plan = compress(
            forward, backward, residuals, outputs=original_outputs
        )
        backward = optimize_training_graph(backward)
        plan["shape_boundaries"] = shape_plan
    if not checkpoint:
        forward = optimize_training_graph(forward)
    plan["retained_values"] = list(residuals)
    positions = {node: index for index, node in enumerate(backward.graph)}
    backward_values = ir.convenience.create_value_mapping(backward.graph)
    plan["release_points"] = []
    for node in plan["nodes"]:
        for key in node["outputs"]:
            value = backward_values.get(key)
            if value is None:
                continue
            consumers = [positions.get(use.node) for use in value.uses()]
            plan["release_points"].append(
                {
                    "value": key,
                    "last_consumer_node_index": max(consumers, default=None)
                    if all(index is not None for index in consumers)
                    else None,
                    "returned_gradient": value in backward.graph.outputs,
                    "managed_by": "ONNX runtime graph execution",
                }
            )
    seeds = row["seeds"]
    # The captured checkpoint VJP supplies the same original inputs and cotangents.
    forward_feeds = {value.name: feeds[value.name] for value in forward.graph.inputs}
    supplied = {
        value.name: feeds[value.name]
        for value in backward.graph.inputs
        if value.name not in residuals
    }
    if not set(seeds) <= supplied.keys():
        raise ValueError("Captured VJP cotangents differ from the split graph")
    names = list(dict.fromkeys([*(value.name for value in source.graph.outputs), *residuals]))
    gradients = [value.name for value in backward.graph.outputs]
    if gradients != block["output_names"]:
        raise ValueError("Split gradients differ from the captured interface")
    with tempfile.TemporaryDirectory(prefix="onnax-block-recompute-") as temporary:
        before = time.perf_counter()
        forward_path = os.path.join(temporary, "forward.onnx")
        backward_path = os.path.join(temporary, "backward.onnx")
        save_graph(forward, forward_path, data_name="forward.data")
        save_graph(backward, backward_path, data_name="backward.data")
        onnx.checker.check_model(forward_path, full_check=True)
        onnx.checker.check_model(backward_path, full_check=True)
        fwd = factory(forward_path)
        bwd = factory(backward_path)
        creation = time.perf_counter() - before

        def execute():
            begin = time.perf_counter()
            primal = dict(zip(names, fwd.run(names, forward_feeds), strict=True))
            boundary = time.perf_counter()
            saved = {key: primal[key] for key in residuals}
            outputs = bwd.run(gradients, {**supplied, **saved})
            finish = time.perf_counter()
            return (
                outputs,
                sum(value.nbytes for value in saved.values()),
                (boundary - begin, finish - boundary, finish - begin),
            )

        outputs, saved_bytes, _ = execute()
        checks = check_outputs(gradients, outputs, expected)
        del outputs
        for _ in range(warmup):
            execute()
        if memory is not None:
            memory.reset_peak()
            baseline_memory = memory.snapshot()
        measurements = []
        for index in range(samples):
            outputs, _, timing = execute()
            measurements.append(timing)
            if index == samples - 1:
                final_checks = check_outputs(gradients, outputs, expected)
            del outputs
        if arrays_record(feeds) != block["input_arrays"]:
            raise AssertionError("Runtime mutated a captured input")
        allocator = memory.snapshot() if memory is not None else None
        timings = np.array(measurements)
        result = {
            "version": 1,
            "block": name,
            "bundle_identity": record["identity"],
            "input_arrays": block["input_arrays"],
            "plan": plan,
            "retained_activation_bytes": saved_bytes,
            "gradient_checks": checks,
            "final_gradient_checks": final_checks,
            "warmup": warmup,
            "samples": samples,
            "session_creation_seconds": creation,
            "timing_seconds": {
                key: {
                    "median": float(np.median(timings[:, index])),
                    "p95": float(np.percentile(timings[:, index], 95)),
                }
                for index, key in enumerate(("forward", "backward", "paired"))
            },
            "allocator_before": baseline_memory if memory is not None else None,
            "allocator_after": allocator,
            "window_allocator_peak_bytes": None
            if memory is None
            else max(baseline_memory["active_bytes"], allocator["peak_bytes"]),
            "physical_memory": physical_footprint(),
            "timing_scope": "Paired split block forward and backward, including runtime exchange and saved-array scheduling; fixed parameters; excludes other blocks, task loss and optimizer",
            "memory_scope": "One worker, source and derivative graphs, captured CPU arrays and two runtime sessions; allocator peak reset after warmup",
        }
        fwd = bwd = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--captures", required=True)
    parser.add_argument("--block", required=True)
    strategies = parser.add_mutually_exclusive_group()
    strategies.add_argument("--recompute", nargs="+", default=[])
    strategies.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--save-shapes", action="store_true")
    parser.add_argument("--attention-vjp", action="store_true")
    parser.add_argument("--recompute-attention", action="store_true")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.provider == "mlx":
        if not args.library:
            parser.error("MLX requires --library")
        from examples.lora.mlx_runtime import register_library, session

        register_library(args.library)
        memory = MlxMemory(args.library)
    else:
        if args.library:
            parser.error("--library requires MLX")
        from examples.lora.train import session

        memory = None
    result = measure(
        args.bundle,
        args.captures,
        args.block,
        recompute=args.recompute,
        checkpoint=args.checkpoint,
        save_shapes=args.save_shapes,
        attention_vjp=args.attention_vjp,
        recompute_attention=args.recompute_attention,
        factory=session,
        memory=memory,
        warmup=args.warmup,
        samples=args.samples,
    )
    result.update(
        provider=args.provider,
        cpu_fallback_disabled=args.provider == "mlx",
        ort_graph_optimization="disabled",
        library_sha256=None if args.library is None else digest(args.library),
        platform=platform.platform(),
        runtime_environment={
            key: os.environ.get(key)
            for key in (
                "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                "ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES",
                "ONNXRUNTIME_EP_MLX_TRACE",
            )
        },
        versions={
            key: version(key)
            for key in ("onnx", "onnx-ir", "onnxscript", "onnxruntime", "numpy")
        },
        source_sha256={
            os.path.basename(path): digest(path)
            for path in (
                __file__,
                os.path.join(os.path.dirname(__file__), "residuals.py"),
                os.path.join(os.path.dirname(__file__), "recomputation.py"),
                os.path.join(os.path.dirname(__file__), "shape_residuals.py"),
                os.path.join(os.path.dirname(__file__), "optimization.py"),
                os.path.join(os.path.dirname(__file__), "attention_vjp.py"),
            )
        },
    )
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
