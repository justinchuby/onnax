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
    save_activations=False,
    mlx_memory_limit_bytes=None,
    torch_device="cpu",
    torch_chunked=False,
    torch_attention="eager",
    progress_every=0,
    fused_ops=False,
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
    if save_activations and mode != "chunked":
        raise ValueError("Explicit saved activations require chunked mode")
    if mlx_memory_limit_bytes is not None:
        if (
            isinstance(mlx_memory_limit_bytes, bool)
            or not isinstance(mlx_memory_limit_bytes, int)
            or mlx_memory_limit_bytes < 1
        ):
            raise ValueError("MLX allocator limit must be a positive integer")
        if provider != "mlx" or not library:
            raise ValueError("An MLX allocator limit requires the mlx provider and library")
        from examples.huggingface.long_context import memory_limit

        memory_limit(library, mlx_memory_limit_bytes)
    if fused_ops and (mode == "pytorch" or provider == "cpu"):
        raise ValueError("Fused ONNX operations require the MLX or reference provider")
    if torch_device not in ("cpu", "mps"):
        raise ValueError("Expected cpu or mps PyTorch device")
    if torch_attention not in ("eager", "sdpa"):
        raise ValueError("Expected eager or sdpa PyTorch attention")
    if mode != "pytorch" and (
        torch_device != "cpu" or torch_chunked or torch_attention != "eager"
    ):
        raise ValueError("PyTorch device/head options require pytorch mode")
    if torch_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("The PyTorch MPS device is unavailable")
    if checkpoint_layers and mode not in ("chunked", "pytorch"):
        raise ValueError("Decoder checkpoints require chunked or pytorch mode")
    if projection_precision == "bf16" and (
        mode not in ("pytorch", "joint", "chunked")
        or (mode != "pytorch" and provider != "mlx")
    ):
        raise ValueError("BF16 measurement uses PyTorch or MLX joint/chunked mode")
    torch.set_num_threads(1)
    session_totals = {}
    start = time.perf_counter()
    if mode == "pytorch":
        model = reference_model(real)
        model.set_attn_implementation(torch_attention)
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
        if checkpoint_layers:
            from examples.huggingface.torch_context import checkpoint_qwen

            model = checkpoint_qwen(model, checkpoint_layers)
        model.to(torch_device)
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

        selected_factory = factory

        class TimedSession:
            def __init__(self, source):
                self.executor = selected_factory(source)
                self.name = os.path.basename(source) if isinstance(source, str) else "body"

            def run(self, *args, **kwargs):
                before = time.perf_counter()
                result = self.executor.run(*args, **kwargs)
                session_totals[self.name] = (
                    session_totals.get(self.name, 0.0) + time.perf_counter() - before
                )
                return result

            def __getattr__(self, name):
                return getattr(self.executor, name)

        model, reference = prepare(
            real,
            mode=mode if mode in ("chunked", "joint") else "full",
            optimize=mode != "raw",
            chunk_size=chunk_size,
            checkpoint_layers=checkpoint_layers,
            save_activations=mode == "saved" or save_activations,
            nf4=nf4,
            projection_precision=projection_precision,
            session_factory=TimedSession,
            fused_ops=fused_ops,
        )
        del reference
    from examples.huggingface.torch_context import synchronize

    device = torch_device if mode == "pytorch" else "cpu"
    synchronize(device)
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
    input_digest = hashlib.sha256(batch["input_ids"].numpy().tobytes()).hexdigest()
    batch = {name: value.to(device) for name, value in batch.items()}
    labels = labels.to(device)
    synchronize(device)
    records = []
    losses = []
    for index in range(warmup + samples):
        sessions_before = session_totals.copy()
        synchronize(device)
        before = time.perf_counter()
        cpu_before = time.process_time()
        update.zero_grad(set_to_none=True)
        if mode == "pytorch":
            if torch_chunked:
                from examples.huggingface.torch_context import chunked_token_loss

                loss = chunked_token_loss(model, batch, labels, chunk_size)
            else:
                prediction = model(**batch, use_cache=False).logits
                loss = token_loss(prediction, labels)
        else:
            loss = model(**batch, labels=labels).loss
        synchronize(device)
        forward = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward = time.perf_counter()
        update.step()
        synchronize(device)
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
            if mode != "pytorch":
                durations = {
                    name: value - sessions_before.get(name, 0.0)
                    for name, value in session_totals.items()
                }
                records[-1]["onnx_execution"] = sum(durations.values())
                if mode == "chunked":
                    records[-1]["head_loss_gradient_execution"] = sum(
                        value for name, value in durations.items() if name == "gradient.onnx"
                    )
                    records[-1]["body_execution"] = (
                        records[-1]["onnx_execution"]
                        - records[-1]["head_loss_gradient_execution"]
                    )
        if progress_every and (
            (index + 1) % progress_every == 0 or index + 1 == warmup + samples
        ):
            print(
                f"benchmark step {index + 1}/{warmup + samples}: "
                f"loss={losses[-1]:.6g}, step={finished - before:.3f}s",
                file=sys.stderr,
                flush=True,
            )
    report = {
        "mode": mode,
        "checkpoint_layers": checkpoint_layers,
        "save_activations": mode == "saved" or save_activations,
        "mlx_memory_limit_bytes": mlx_memory_limit_bytes,
        "provider": "pytorch_" + torch_device if mode == "pytorch" else provider,
        "torch_chunked_head": torch_chunked,
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "attention_implementation": torch_attention
        if mode == "pytorch"
        else ("fused" if fused_ops else "decomposed"),
        "fused_ops": fused_ops,
        "synchronization": "MPS at each phase boundary"
        if device == "mps"
        else "synchronous CPU/ORT execution",
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
        "input_ids_sha256": input_digest,
        "input_pattern": (
            "Prefix of the first naturally long corpus example"
            if train_file is not None
            else "Repeated prefix of the first eight local corpus lines"
        ),
        "head_chunk_size": chunk_size if mode == "chunked" or torch_chunked else None,
        "full_logits_bytes": batch_size * sequence_length * model.config.vocab_size * 4,
        "chunk_logits_bytes": min(chunk_size, tokens) * model.config.vocab_size * 4
        if mode == "chunked" or torch_chunked
        else None,
        "initialization_seconds": initialized - start,
        "timings_ms": {
            key: {
                "median": float(np.median([r[key] for r in records]) * 1000),
                "p95": float(np.percentile([r[key] for r in records], 95) * 1000),
            }
            for key in records[0]
        },
        "timing_samples_ms": [
            {key: value * 1000 for key, value in row.items()} for row in records
        ],
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
    from examples.process_memory import physical_footprint

    report["physical_footprint"] = physical_footprint()
    if device == "mps":
        report["mps_allocator"] = {
            "active_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
            "scope": "Allocator snapshot after synchronized final update; process peak is reported separately",
        }
    if provider == "mlx" and mode != "pytorch":
        import ctypes

        if library:
            api = ctypes.CDLL(library)
            report["mlx_allocator"] = {}
            for name in ("active", "cache", "peak"):
                fn = getattr(api, f"mlx_get_{name}_memory")
                fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int
                value = ctypes.c_size_t()
                if fn(ctypes.byref(value)) != 0:
                    raise RuntimeError("MLX memory query failed")
                report["mlx_allocator"][name + "_bytes"] = value.value
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
    parser.add_argument("--torch-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--torch-chunked", action="store_true")
    parser.add_argument("--fused-ops", action="store_true")
    parser.add_argument("--torch-attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--checkpoint-layers", type=int, default=0)
    parser.add_argument("--train-file")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--mlx-memory-limit-gib", type=int)
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--library", help="Development MLX EP library")
    parser.add_argument("--segment-nodes", type=int)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Report progress to stderr; zero disables it",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", default="qwen-benchmark.json")
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("Require a nonnegative progress interval")
    if args.samples < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("Require positive samples/batch and nonnegative warmup")
    if args.sequence_length < 2 or args.chunk_size < 1:
        parser.error("Require sequence length >= 2 and positive chunk size")
    if not args.real and args.sequence_length > 64:
        parser.error(
            "Tiny Qwen context is limited to 64 tokens; use --real for longer contexts"
        )
    if args.mlx_memory_limit_gib is not None and args.mlx_memory_limit_gib < 1:
        parser.error("Require a positive MLX allocator limit")
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
                    torch_device=args.torch_device,
                    torch_chunked=args.torch_chunked,
                    torch_attention=args.torch_attention,
                    progress_every=args.progress_every,
                    fused_ops=args.fused_ops,
                    checkpoint_layers=args.checkpoint_layers,
                    train_file=args.train_file,
                    save_activations=args.save_activations,
                    mlx_memory_limit_bytes=(
                        args.mlx_memory_limit_gib * 1024**3
                        if args.mlx_memory_limit_gib
                        else None
                    ),
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
        if args.checkpoint_layers or args.save_activations:
            modes = ("chunked",)
        if args.torch_chunked or args.torch_device != "cpu" or args.torch_attention != "eager":
            modes = ("pytorch",)
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
                "--progress-every",
                str(args.progress_every),
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
            if mode == "pytorch":
                cmd.extend(
                    [
                        "--torch-device",
                        args.torch_device,
                        "--torch-attention",
                        args.torch_attention,
                    ]
                )
                if args.torch_chunked:
                    cmd.append("--torch-chunked")
            if args.fused_ops and mode != "pytorch":
                cmd.append("--fused-ops")
            if args.checkpoint_layers:
                cmd.extend(["--checkpoint-layers", str(args.checkpoint_layers)])
            if args.save_activations:
                cmd.append("--save-activations")
            if args.mlx_memory_limit_gib:
                cmd.extend(["--mlx-memory-limit-gib", str(args.mlx_memory_limit_gib)])
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
