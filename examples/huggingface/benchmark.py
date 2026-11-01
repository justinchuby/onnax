# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Isolated PyTorch and ONNX training measurements."""

import argparse
import functools
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
    sequence_length=64,
    chunk_size=128,
    nf4=False,
    provider="cpu",
    library=None,
    segment_nodes=None,
    projection_precision="fp32",
    checkpoint_layers=0,
    train_file=None,
):
    for name, value, minimum in (
        ("sequence_length", sequence_length, 2),
        ("chunk_size", chunk_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not real and sequence_length > 64:
        raise ValueError(
            "Tiny Qwen context is limited to 64 tokens; use --real for longer contexts"
        )
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 projection precision")
    if checkpoint_layers and mode != "chunked":
        raise ValueError("Decoder checkpoints require chunked mode")
    if projection_precision == "bf16" and (
        mode not in ("pytorch", "joint", "chunked")
        or (mode != "pytorch" and provider != "mlx")
    ):
        raise ValueError("BF16 measurement uses PyTorch or MLX joint/chunked mode")
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
        if projection_precision == "bf16":
            from examples.huggingface.mixed_precision import reference_projections

            model = reference_projections(model)
    else:
        from examples.lora.train import session as cpu_session

        factory = cpu_session
        native_construction = {}
        if provider == "mlx":
            from examples.lora.mlx_runtime import register_library
            from examples.lora.mlx_runtime import session as mlx_session

            if library:
                register_library(library)
            selected = functools.partial(mlx_session, target_nodes=segment_nodes)

            def factory(source):
                before = time.perf_counter()
                executor = selected(source)
                native_construction[f"session_{len(native_construction)}"] = {
                    "factory_seconds": time.perf_counter() - before,
                    "segments": len(executor.segments) if segment_nodes is not None else 1,
                    "source": "protobuf bytes"
                    if isinstance(source, bytes)
                    else os.path.basename(source),
                }
                return executor

        model, reference = prepare(
            real,
            mode=mode if mode in ("chunked", "joint") else "full",
            optimize=mode != "raw",
            chunk_size=chunk_size,
            checkpoint_layers=checkpoint_layers,
            save_activations=mode == "saved",
            nf4=nf4,
            projection_precision=projection_precision,
            session_factory=factory,
        )
        del reference
    params = [p for p in model.parameters() if p.requires_grad]
    update = torch.optim.AdamW(params, lr=0.003, weight_decay=0)
    initialized = time.perf_counter()
    tok = tokenizer(real)
    if sequence_length > model.config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model context limit")
    if train_file is not None:
        source_ids = next(
            (
                ids
                for text in texts(train_file)
                if len(ids := tok(text)["input_ids"]) >= sequence_length
            ),
            None,
        )
        if source_ids is None:
            raise ValueError("No corpus example reaches the requested benchmark length")
    else:
        source_ids = tok(" ".join(texts()[:8]), truncation=True, max_length=sequence_length)[
            "input_ids"
        ]
    if not source_ids:
        raise ValueError("Benchmark corpus must tokenize to at least one token")
    ids = (source_ids * ((sequence_length + len(source_ids) - 1) // len(source_ids)))[
        :sequence_length
    ]
    batch = {
        "input_ids": torch.tensor([ids] * batch_size, dtype=torch.int64),
        "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.int64),
    }
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    tokens = int(labels[:, 1:].ne(-100).sum())
    records = []
    losses = []
    for index in range(warmup + samples):
        before = time.perf_counter()
        cpu_before = time.process_time()
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
                    "loss_and_gradients" if mode == "joint" else "forward_and_loss": forward
                    - before,
                    "gradient_scaling" if mode == "joint" else "backward": backward - forward,
                    "process_cpu": time.process_time() - cpu_before,
                    "adamw": finished - backward,
                    "step": finished - before,
                }
            )
    report = {
        "mode": mode,
        "checkpoint_layers": checkpoint_layers,
        "provider": "pytorch_cpu" if mode == "pytorch" else provider,
        "segment_target_nodes": segment_nodes
        if mode != "pytorch" and provider == "mlx"
        else None,
        "lut4_fusion_disabled": os.environ.get("ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION") == "1",
        "mlx_compile_disabled": os.environ.get("ONNXRUNTIME_EP_MLX_NO_COMPILE", "")
        not in ("", "0"),
        "real": real,
        "weight_values": "nf4_reconstructed_fp32" if nf4 else "fp32",
        "projection_precision": projection_precision,
        "gradient_and_optimizer_dtype": "float32",
        "base_storage": "packed_nf4_decoder" if nf4 and mode != "pytorch" else "fp32",
        "warmup": warmup,
        "samples": samples,
        "losses": losses,
        "batch": batch_size,
        "sequence": batch["input_ids"].shape[1],
        "valid_shifted_tokens": tokens,
        "input_ids_sha256": hashlib.sha256(batch["input_ids"].numpy().tobytes()).hexdigest(),
        "input_pattern": (
            "Prefix of the first naturally long corpus example"
            if train_file is not None
            else "Repeated prefix of the first eight local corpus lines"
        ),
        "head_chunk_size": chunk_size if mode == "chunked" else None,
        "full_logits_bytes": batch_size * sequence_length * model.config.vocab_size * 4,
        "chunk_logits_bytes": min(chunk_size, tokens) * model.config.vocab_size * 4
        if mode == "chunked"
        else None,
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
            else "Model construction, differentiation, export and selected execution sessions"
        ),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "threads": {"torch": torch.get_num_threads(), "ort_intra": 1, "ort_inter": 1},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
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
        if checkpoint_layers:
            executed_graphs = {}
            for execution in model.onnx.executions:
                executed_graphs[execution.block.name + "_forward"] = execution.block.model
                if execution.derivative is not None:
                    executed_graphs[execution.block.name + "_backward"] = execution.derivative
        else:
            executed_graphs = (
                {"loss_gradient": model.onnx.gradient}
                if mode == "joint"
                else {"forward": model.onnx.model, "backward": model.onnx.optimized_vjp}
            )
        if mode == "chunked":
            executed_graphs["head_backward"] = model.head.gradient
            report["head_forward_session_created"] = model.head.forward_executor is not None
            if model.head.forward_executor is not None:
                executed_graphs["head_forward"] = model.head.model
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
        report["nodes"] = (
            {name: len(graph.graph) for name, graph in executed_graphs.items()}
            if checkpoint_layers
            else (
                {
                    "forward": len(model.forward_graph.graph),
                    "loss_gradient": len(model.onnx.gradient.graph),
                }
                if mode == "joint"
                else {
                    "forward": len(model.onnx.model.graph),
                    "raw_vjp": len(model.onnx.vjp_model.graph),
                    "executed_vjp": len(model.onnx.optimized_vjp.graph),
                }
            )
        )
        report["logical_saved_activation_bytes"] = getattr(
            model.onnx, "last_residual_bytes", 0
        )
        report["saved_activation_count"] = (
            0 if mode == "joint" or checkpoint_layers else len(model.onnx.residual_names)
        )
        report["input_view_median_us"] = float(
            np.median(model.onnx.timings["exchange"][warmup:]) * 1e6
        )
        report["input_view_scope"] = (
            "Host data/adapter preparation and validation; runtime transfers are included in execution"
            if mode == "joint"
            else "Host data-input preparation; adapter preparation is included in forward"
        )
        if mode == "joint":
            report["autograd_saved_gradient_bytes"] = model.onnx.last_saved_gradient_bytes
            report["public_session_payload_bytes"] = {
                "inputs": model.onnx.last_input_bytes,
                "outputs": model.onnx.last_output_bytes,
                "scope": "Logical NumPy session arguments and results; excludes internal EP transfers",
            }
            durations = np.asarray(model.onnx.timings["loss_gradient"][warmup:]) * 1000
            report["session_execution_ms"] = {
                "median": float(np.median(durations)),
                "p95": float(np.percentile(durations, 95)),
            }
            report["timing_scope"] = (
                "Forward computes loss and gradients; PyTorch backward scales returned gradients"
            )

        report["exchange_scope"] = (
            "CPU NumPy input views; input snapshots and parameter/output views are included in execution boundaries"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["pytorch", "raw", "optimized", "saved", "chunked", "joint"]
    )
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--checkpoint-layers", type=int, default=0)
    parser.add_argument("--train-file")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library", help="Development MLX EP library")
    parser.add_argument("--segment-nodes", type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", default="qwen-benchmark.json")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("Require positive samples/batch and nonnegative warmup")
    if args.sequence_length < 2 or args.chunk_size < 1:
        parser.error("Require sequence length >= 2 and positive chunk size")
    if not args.real and args.sequence_length > 64:
        parser.error(
            "Tiny Qwen context is limited to 64 tokens; use --real for longer contexts"
        )
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
                    checkpoint_layers=args.checkpoint_layers,
                    train_file=args.train_file,
                    samples=args.samples,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    chunk_size=args.chunk_size,
                    nf4=args.nf4,
                    provider=args.provider,
                    library=args.library,
                    segment_nodes=args.segment_nodes,
                    projection_precision=args.projection_precision,
                )
            )
        )
    else:
        reports = []
        modes = (
            ("pytorch", "joint")
            if args.projection_precision == "bf16"
            else ("pytorch", "raw", "optimized", "saved", "chunked", "joint")
        )
        if args.checkpoint_layers:
            modes = ("chunked",)
        for mode in modes:
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
                "--sequence-length",
                str(args.sequence_length),
                "--chunk-size",
                str(args.chunk_size),
                "--provider",
                args.provider,
                "--projection-precision",
                args.projection_precision,
            ]
            if args.checkpoint_layers:
                cmd.extend(["--checkpoint-layers", str(args.checkpoint_layers)])
            if args.train_file:
                cmd.extend(["--train-file", args.train_file])
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
