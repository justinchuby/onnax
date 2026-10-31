# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare NF4 Qwen training quality across runtimes and independent seeds."""

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import platform
import sys

import numpy as np
import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.mixed_precision import reference_projections
from examples.huggingface.qwen import REVISION
from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.reference_losses import stable_token_loss
from examples.huggingface.sft import OnnxQwen3, _prepare_graph, texts, tokenizer
from examples.huggingface.validation import add_reference_parameters

ENGINES = ("torch_cpu", "torch_cuda", "onnx_reference", "mlx")


def _digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _batch(tok, lines):
    encoded = tok(lines, padding=True, truncation=True, max_length=64, return_tensors="pt")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    labels = ids.clone().masked_fill(mask == 0, -100)
    return dict(
        input_ids=ids,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
        labels=labels,
    )


def _loss(model, batch, count, engine):
    if engine.startswith("torch_"):
        device = next(model.parameters()).device
        batch = {name: value.to(device) for name, value in batch.items()}
        inputs = {name: value for name, value in batch.items() if name != "labels"}
        logits = model(**inputs, use_cache=False).logits
        return stable_token_loss(logits, batch["labels"], count)
    return model(**batch, num_items_in_batch=count).loss


def _evaluate(model, batches, engine):
    count = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in batches)
    with torch.no_grad():
        result = sum(float(_loss(model, batch, count, engine).detach()) for batch in batches)
    if not np.isfinite(result):
        raise AssertionError("Nonfinite evaluation loss")
    return result


def _schedule(seed, size, steps, microbatch, accumulation):
    rng, order = np.random.default_rng(seed), []
    total = steps * microbatch * accumulation
    while len(order) < total:
        order.extend(rng.permutation(size).tolist())
    return np.asarray(order[:total], dtype=np.int64).reshape(steps, accumulation, microbatch)


def _build(engine, real, precision, library):
    reference, graph, params, mapping, quantization = _prepare_graph(
        real, nf4=True, projection_precision=precision
    )
    base = hashlib.sha256(
        json.dumps(frozen_hashes(graph), sort_keys=True).encode()
    ).hexdigest()
    if engine.startswith("torch_"):
        model = add_reference_parameters(reference, params, mapping)
        if precision == "bf16":
            model = reference_projections(model)
        model = model.to("cuda" if engine == "torch_cuda" else "cpu")
        parameters = {
            row[key]: getattr(model.get_submodule(row["module"]), key)
            for row in mapping
            for key in ("A", "B")
        }
    else:
        if engine == "mlx":
            from examples.lora.mlx_runtime import register_library, session

            if library is not None:
                register_library(library)
            factory = session
        else:
            from examples.huggingface.serialization import reference_session

            factory = reference_session
        model = OnnxQwen3(
            reference.config, graph, params, mapping, mode="joint", session_factory=factory
        )
        model.quantization, model.projection_precision = quantization, precision
        parameters = dict(zip(model.onnx.names, model.onnx.adapters, strict=True))
    return model, dict(sorted(parameters.items())), base


def _frozen(model, engine):
    if engine.startswith("torch_"):
        return _digest(
            {
                name: value.detach().cpu().numpy()
                for name, value in model.named_parameters()
                if not value.requires_grad
            }
        )
    return hashlib.sha256(
        json.dumps(frozen_hashes(model.forward_graph), sort_keys=True).encode()
    ).hexdigest()


def _trial(
    model,
    parameters,
    engine,
    seed,
    train,
    heldout,
    tok,
    *,
    steps,
    microbatch,
    accumulation,
    interval,
):
    rng = np.random.default_rng(seed)
    initial = {
        name: rng.normal(0, 0.02, tuple(value.shape)).astype(np.float32)
        if name.endswith(".A")
        else np.zeros(tuple(value.shape), np.float32)
        for name, value in parameters.items()
    }
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(torch.from_numpy(initial[name]).to(value.device))
    optimizer = torch.optim.AdamW(
        parameters.values(), lr=0.003, eps=1e-8, weight_decay=0, foreach=False
    )
    order = _schedule(seed, len(train), steps, microbatch, accumulation)
    evaluation = [
        _batch(tok, heldout[start : start + microbatch])
        for start in range(0, len(heldout), microbatch)
    ]
    training_evaluation = [
        _batch(tok, train[start : start + microbatch])
        for start in range(0, len(train), microbatch)
    ]
    curves, losses, snapshots, norms = [], [], {}, []
    initial_train = _evaluate(model, training_evaluation, engine)
    curves.append({"step": 0, "loss": _evaluate(model, evaluation, engine)})
    for step, indices in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        batches = [_batch(tok, [train[i] for i in group]) for group in indices]
        count = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in batches)
        total = 0.0
        for batch in batches:
            loss = _loss(model, batch, count, engine)
            total += float(loss.detach())
            loss.backward()
        vector = torch.cat([value.grad.detach().reshape(-1) for value in parameters.values()])
        if not torch.isfinite(vector).all() or not np.isfinite(total):
            raise AssertionError("Nonfinite training loss or gradient")
        norms.append(float(vector.norm()))
        if step == 1 or step % interval == 0 or step == steps:
            snapshots[step] = vector.cpu().numpy().copy()
        optimizer.step()
        losses.append(total)
        if step == 1 or step % max(1, interval // 4) == 0 or step == steps:
            print(
                json.dumps({"engine": engine, "seed": seed, "step": step, "loss": total}),
                file=sys.stderr,
                flush=True,
            )
        if step % interval == 0 or step == steps:
            curves.append({"step": step, "loss": _evaluate(model, evaluation, engine)})
    final = (
        torch.cat([value.detach().reshape(-1) for value in parameters.values()])
        .cpu()
        .numpy()
        .copy()
    )
    report = {
        "engine": engine,
        "seed": seed,
        "initial_adapter_sha256": _digest(initial),
        "schedule_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "training_loss": losses,
        "gradient_norm": norms,
        "validation_curve": curves,
        "initial_training_loss": initial_train,
        "final_training_loss": _evaluate(model, training_evaluation, engine),
        "initial_validation_loss": curves[0]["loss"],
        "final_validation_loss": curves[-1]["loss"],
    }
    print(
        json.dumps(
            {
                "engine": engine,
                "seed": seed,
                "final_validation_loss": report["final_validation_loss"],
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    return report, snapshots, final


def _vector_metrics(a, b):
    x, y = a.astype(np.float64), b.astype(np.float64)
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    return {
        "cosine": float(np.dot(x, y) / (nx * ny)) if nx and ny else float(nx == ny),
        "relative_l2": float(np.linalg.norm(x - y) / max(ny, 1e-30)),
        "max_abs": float(np.max(np.abs(x - y))),
    }


def run(
    *,
    engines=ENGINES,
    seeds=(17, 29, 43, 71, 101),
    steps=100,
    real=False,
    precision="bf16",
    library=None,
    microbatch=1,
    accumulation=4,
    interval=20,
    capture_path=None,
):
    if not engines or len(set(engines)) != len(engines) or set(engines) - set(ENGINES):
        raise ValueError("Select unique supported engines")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or min(steps, microbatch, accumulation, interval) < 1
    ):
        raise ValueError("Expected unique nonnegative seeds and positive run dimensions")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    available = {
        "torch_cpu": True,
        "onnx_reference": True,
        "torch_cuda": torch.cuda.is_available(),
        "mlx": sys.platform == "darwin"
        and importlib.util.find_spec("onnxruntime_ep_mlx") is not None,
    }
    if available["torch_cuda"] and precision == "bf16":
        available["torch_cuda"] = torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    selected = [engine for engine in engines if available[engine]]
    if not selected:
        raise ValueError("No requested engine is available")
    tok, corpus = tokenizer(real), texts()
    split = max(1, len(corpus) * 3 // 4)
    train, heldout = corpus[:split], corpus[split:]
    if not heldout:
        raise ValueError("A held-out split needs at least two corpus lines")
    reports, captures, bases = [], {}, {}
    for engine in selected:
        print(json.dumps({"engine": engine, "stage": "build"}), file=sys.stderr, flush=True)
        model, parameters, base = _build(engine, real, precision, library)
        parameter_order = list(parameters)
        frozen = _frozen(model, engine)
        context = (
            pairwise_head(model, projection_precision=precision)
            if engine.startswith("torch_")
            else contextlib.nullcontext()
        )
        with context:
            for seed in seeds:
                report, gradients, final = _trial(
                    model,
                    parameters,
                    engine,
                    seed,
                    train,
                    heldout,
                    tok,
                    steps=steps,
                    microbatch=microbatch,
                    accumulation=accumulation,
                    interval=interval,
                )
                reports.append(report)
                captures[engine, seed] = (gradients, final)
        if _frozen(model, engine) != frozen:
            raise AssertionError("Frozen base changed during training")
        bases[engine] = base
        del model, parameters
        gc.collect()
    if len(set(bases.values())) != 1:
        raise AssertionError("Engines used different frozen source weights")
    comparisons = []
    for left, right in itertools.combinations(selected, 2):
        for seed in seeds:
            a, b = [
                next(row for row in reports if row["engine"] == engine and row["seed"] == seed)
                for engine in (left, right)
            ]
            if (
                a["initial_adapter_sha256"] != b["initial_adapter_sha256"]
                or a["schedule_sha256"] != b["schedule_sha256"]
            ):
                raise AssertionError("Paired runs must share initialization and data order")
            ga, pa = captures[left, seed]
            gb, pb = captures[right, seed]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "seed": seed,
                    "final_validation_loss_gap": a["final_validation_loss"]
                    - b["final_validation_loss"],
                    "parameters": _vector_metrics(pa, pb),
                    "gradient_snapshots": [
                        {"step": step, **_vector_metrics(ga[step], gb[step])} for step in ga
                    ],
                }
            )
    summaries = {}
    for engine in selected:
        values = [row["final_validation_loss"] for row in reports if row["engine"] == engine]
        summaries[engine] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        }
    capture = None
    if capture_path is not None:
        arrays = {}
        for (engine, seed), (gradients, parameters) in captures.items():
            arrays[f"{engine}__{seed}__parameters"] = parameters
            arrays.update(
                {
                    f"{engine}__{seed}__gradient_{step}": value
                    for step, value in gradients.items()
                }
            )
        with open(capture_path, "wb") as stream:
            np.savez(stream, **arrays)
        with open(capture_path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        capture = {"version": 1, "sha256": digest, "parameter_order": parameter_order}
    library_hash = None
    if library is not None and "mlx" in selected:
        with open(library, "rb") as stream:
            library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    versions = {}
    for name in (
        "onnax",
        "torch",
        "numpy",
        "onnxruntime",
        "onnxscript",
        "transformers",
        "mobius-onnx",
        "onnxruntime-ep-mlx",
        "mlx",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "nf4": True,
        "revision": REVISION if real else None,
        "base_seed": 20261029,
        "projection_precision": precision,
        "library_sha256": library_hash,
        "mlx_compile_disabled": os.environ.get("ONNXRUNTIME_EP_MLX_NO_COMPILE") == "1",
        "capture": capture,
        "engines": selected,
        "unavailable_engines": [engine for engine in engines if not available[engine]],
        "seeds": list(seeds),
        "steps": steps,
        "microbatch": microbatch,
        "accumulation": accumulation,
        "evaluation_interval": interval,
        "optimizer": {
            "name": "AdamW",
            "lr": 0.003,
            "eps": 1e-8,
            "weight_decay": 0,
            "foreach": False,
        },
        "precision_policy": "FP32 adapters, gradients, Adam; BF16 projection operands/results"
        if precision == "bf16"
        else "FP32 computation and state",
        "reference_head": "FP32 pairwise K=128; stable target-excluded CE",
        "frozen_unchanged": True,
        "source_base_sha256": next(iter(bases.values())),
        "corpus_sha256": hashlib.sha256("\n".join(corpus).encode()).hexdigest(),
        "train_lines": len(train),
        "heldout_lines": len(heldout),
        "gradient_scope": "Independent trajectory states before indicated update; identical batch schedule for each seed",
        "measurement_scope": "Training-quality experiment with host snapshots for gradient comparison",
        "ort_optimizer": "disabled",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if "torch_cuda" in selected and precision == "bf16"
        else None,
        "platform": platform.platform(),
        "versions": versions,
        "hardware": {
            "cpu_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "cuda_device": torch.cuda.get_device_name(0) if "torch_cuda" in selected else None,
        },
        "validation_summary": summaries,
        "comparisons": comparisons,
        "trials": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=ENGINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43, 71, 101])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--library")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument(
        "--capture", help="Save FP32 gradient snapshots and final adapters as NPZ"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        engines=args.engines,
        seeds=args.seeds,
        steps=args.steps,
        real=args.real,
        precision=args.projection_precision,
        library=args.library,
        capture_path=args.capture,
        microbatch=args.microbatch,
        accumulation=args.accumulation,
        interval=args.interval,
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
