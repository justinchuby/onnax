# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Isolated PyTorch/raw ONNX/optimized ONNX/chunked training measurements."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import torch

from examples.huggingface.qwen import reference_model
from examples.huggingface.sft import prepare, texts, token_loss, tokenizer
from examples.huggingface.validation import TorchLora


def native_sessions(model, *, saved, library=None, segment_nodes=None):
    """Replace CPU construction sessions with reusable native graph executors."""
    from examples.huggingface.residuals import split_vjp
    from examples.huggingface.segmented import SegmentedSession
    from examples.huggingface.serialization import save_graph
    from examples.lora.mlx_runtime import register_library, session

    if library:
        register_library(library)
    measurements = {}

    def create(graph, name):
        start = time.perf_counter()
        if segment_nodes is not None:
            executor = SegmentedSession(
                graph, target_nodes=segment_nodes, session_factory=session
            )
        else:
            path = os.path.join(model.onnx.runtime_directory.name, name + ".onnx")
            save_graph(graph, path, data_name=name + ".data")
            executor = session(path)
        measurements[name] = {
            "export_and_session_seconds": time.perf_counter() - start,
            "segments": len(executor.segments) if segment_nodes is not None else 1,
        }
        return executor

    forward = model.onnx.model
    if saved:
        forward, _, _ = split_vjp(forward, model.onnx.vjp_model)
    model.onnx.forward_session = create(forward, "native-forward")
    model.onnx.backward_session = create(model.onnx.optimized_vjp, "native-backward")
    if model.mode == "chunked":
        model.head.executor = create(model.head.gradient, "native-head-gradient")
    return measurements


def hardware():
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        cpu = result.stdout.strip() or cpu
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }


def measure(
    mode,
    *,
    real=False,
    samples=100,
    warmup=10,
    batch_size=4,
    nf4=False,
    provider="cpu",
    library=None,
    segment_nodes=None,
):
    torch.set_num_threads(1)
    start = time.perf_counter()
    if mode == "pytorch":
        model = reference_model(real)
        if nf4:
            from examples.huggingface.nf4 import quantize_reference

            quantize_reference(model)
        for p in model.parameters():
            p.requires_grad_(False)
        count = 0
        for name, layer in list(model.named_modules()):
            if name.endswith((".self_attn.q_proj", ".self_attn.v_proj")):
                parent, child = name.rsplit(".", 1)
                rng = np.random.default_rng(20261029 + count)
                a = rng.normal(0, 0.02, (layer.in_features, 4)).astype(np.float32)
                b = np.zeros((4, layer.out_features), np.float32)
                setattr(model.get_submodule(parent), child, TorchLora(layer, a, b, 0.25))
                count += 2
    else:
        model, reference = prepare(
            real,
            mode="chunked" if mode == "chunked" else "full",
            optimize=mode != "raw",
            save_activations=mode == "saved",
            nf4=nf4,
        )
        del reference
        if provider == "mlx":
            native_construction = native_sessions(
                model, saved=mode == "saved", library=library, segment_nodes=segment_nodes
            )
    params = [p for p in model.parameters() if p.requires_grad]
    update = torch.optim.AdamW(params, lr=0.003, weight_decay=0)
    initialized = time.perf_counter()
    tok = tokenizer(real)
    batch = tok(
        [" ".join(texts()[:8])] * batch_size,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )
    batch = {n: v for n, v in batch.items() if n in ("input_ids", "attention_mask")}
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    tokens = int(labels[:, 1:].ne(-100).sum())
    records = []
    losses = []
    for index in range(warmup + samples):
        before = time.perf_counter()
        update.zero_grad(set_to_none=True)
        if mode == "pytorch":
            prediction = model(**batch, use_cache=False).logits
            loss = token_loss(prediction, labels)
        else:
            loss = model(**batch, labels=labels).loss
        forward = time.perf_counter()
        loss.backward()
        backward = time.perf_counter()
        update.step()
        finished = time.perf_counter()
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"Nonfinite training loss at step {index + 1}")
        if index >= warmup:
            records.append(
                {
                    "forward_and_loss": forward - before,
                    "backward": backward - forward,
                    "adamw": finished - backward,
                    "step": finished - before,
                }
            )
    report = {
        "mode": mode,
        "provider": "pytorch_cpu" if mode == "pytorch" else provider,
        "segment_target_nodes": segment_nodes if mode != "pytorch" else None,
        "lut4_fusion_disabled": os.environ.get("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION") == "1",
        "real": real,
        "weight_values": "nf4_reconstructed_fp32" if nf4 else "fp32",
        "base_storage": "packed_nf4_decoder" if nf4 and mode != "pytorch" else "fp32",
        "warmup": warmup,
        "samples": samples,
        "losses": losses,
        "batch": batch_size,
        "sequence": batch["input_ids"].shape[1],
        "valid_shifted_tokens": tokens,
        "initialization_seconds": initialized - start,
        "timings_ms": {
            key: {
                "median": float(np.median([r[key] for r in records]) * 1000),
                "p95": float(np.percentile([r[key] for r in records], 95) * 1000),
            }
            for key in records[0]
        },
        "effective_tokens_per_second": tokens / float(np.median([r["step"] for r in records])),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "Whole isolated process, including model construction, runtime, arrays and optimizer",
        "initialization_scope": (
            "PyTorch model construction and FP32 base reconstruction"
            if mode == "pytorch"
            else "Model construction, initial CPU sessions, and selected execution sessions"
        ),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "threads": {"torch": 1, "ort_intra": 1, "ort_inter": 1},
        "ort_optimizer": "disabled",
        "platform": platform.platform(),
        "hardware": hardware(),
        "versions": {
            n: version(n)
            for n in ("torch", "transformers", "onnxruntime", "onnxscript", "mobius-onnx")
        },
    }
    if provider == "mlx" and mode != "pytorch":
        report["native_construction"] = native_construction
        report["cpu_fallback_disabled"] = True
        report["library"] = os.path.abspath(library) if library else "installed plugin"
        if library:
            with open(library, "rb") as stream:
                report["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if mode != "pytorch":
        executed_graphs = {"forward": model.onnx.model, "backward": model.onnx.optimized_vjp}
        if mode == "chunked":
            executed_graphs.update(
                head_forward=model.head.model, head_backward=model.head.gradient
            )
        report["initializer_bytes_by_session"] = {
            name: sum(value.const_value.nbytes for value in graph.graph.initializers.values())
            for name, graph in executed_graphs.items()
        }
        report["initializer_scope"] = (
            "Logical top-level initializer bytes for each execution session"
        )
        if nf4:
            report["decoder_storage_bytes"] = {
                "original": model.quantization["original_decoder_bytes"],
                "packed": model.quantization["packed_decoder_bytes"],
            }
        report["nodes"] = {
            "forward": len(model.onnx.model.graph),
            "raw_vjp": len(model.onnx.vjp_model.graph),
            "executed_vjp": len(model.onnx.optimized_vjp.graph),
        }
        report["logical_saved_activation_bytes"] = getattr(
            model.onnx, "last_residual_bytes", 0
        )
        report["saved_activation_count"] = len(model.onnx.residual_names)
        report["input_view_median_us"] = float(
            np.median(model.onnx.timings["exchange"][warmup:]) * 1e6
        )
        report["exchange_scope"] = (
            "CPU NumPy input views; input snapshots and parameter/output views are included in execution boundaries"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pytorch", "raw", "optimized", "saved", "chunked"])
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library", help="Development MLX EP library")
    parser.add_argument("--segment-nodes", type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="qwen-benchmark.json")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("Require positive samples/batch and nonnegative warmup")
    if args.segment_nodes is not None and args.segment_nodes < 1:
        parser.error("Require a positive segment node target")
    if args.provider != "mlx" and (args.library or args.segment_nodes):
        parser.error("--library and --segment-nodes require --provider mlx")
    if args.mode:
        print(
            json.dumps(
                measure(
                    args.mode,
                    real=args.real,
                    samples=args.samples,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    nf4=args.nf4,
                    provider=args.provider,
                    library=args.library,
                    segment_nodes=args.segment_nodes,
                )
            )
        )
    else:
        reports = []
        for mode in ("pytorch", "raw", "optimized", "saved", "chunked"):
            cmd = [
                sys.executable,
                "-m",
                "examples.huggingface.benchmark",
                "--mode",
                mode,
                "--samples",
                str(args.samples),
                "--warmup",
                str(args.warmup),
                "--batch-size",
                str(args.batch_size),
                "--provider",
                args.provider,
            ]
            if args.library:
                cmd.extend(["--library", args.library])
            if args.segment_nodes:
                cmd.extend(["--segment-nodes", str(args.segment_nodes)])
            if args.real:
                cmd.append("--real")
            if args.nf4:
                cmd.append("--nf4")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            reports.append(json.loads(result.stdout))
        with open(args.output, "w") as stream:
            json.dump(reports, stream, indent=2)
        print(os.path.abspath(args.output))
