# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Evaluate captured final adapters and their gradients at shared parameter states."""

import argparse
import contextlib
import hashlib
import json
import sys

import numpy as np
import torch

from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.training_impact import (
    ENGINES,
    _batch,
    _build,
    _evaluate,
    _frozen,
    _loss,
    _schedule,
    _tokenizer_digest,
    texts,
    tokenizer,
)
from examples.huggingface.training_quality_compare import compare


def replay(reports, captures, *, engine, library=None, capture_path, corpus_path=None):
    comparison = compare(reports, captures)
    settings = comparison["settings"]
    if engine not in ENGINES:
        raise ValueError("Unknown replay engine")
    if settings["model"] not in ("tiny Qwen3", "Qwen3-0.6B"):
        raise ValueError("Unknown captured model")
    real = settings["model"] == "Qwen3-0.6B"
    precision = settings["projection_precision"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if engine == "torch_cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA replay requires a CUDA device")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support BF16")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    corpus = texts(corpus_path)
    if hashlib.sha256("\n".join(corpus).encode()).hexdigest() != settings["corpus_sha256"]:
        raise ValueError("Replay corpus differs from capture")
    train, heldout = corpus[: settings["train_lines"]], corpus[settings["train_lines"] :]
    if len(heldout) != settings["heldout_lines"]:
        raise ValueError("Replay data split differs from capture")
    tok = tokenizer(real, training_texts=train if corpus_path is not None else None)
    if settings.get("tokenizer_sha256") not in (None, _tokenizer_digest(tok)):
        raise ValueError("Replay tokenizer differs from capture")
    model, parameters, base = _build(engine, real, precision, library)
    if base != settings["source_base_sha256"]:
        raise ValueError("Replay source weights differ from captured weights")
    expected_order = comparison["components"][0]["capture"]["parameter_order"]
    if list(parameters) != expected_order:
        raise ValueError("Replay parameter order differs from capture")
    microbatch, accumulation = settings["microbatch"], settings["accumulation"]
    evaluation = [
        _batch(tok, heldout[start : start + microbatch])
        for start in range(0, len(heldout), microbatch)
    ]
    frozen, states, gradients = _frozen(model, engine), [], {}
    context = (
        pairwise_head(model, projection_precision=precision)
        if engine.startswith("torch_")
        else contextlib.nullcontext()
    )
    with context:
        for component, capture in zip(comparison["components"], captures, strict=True):
            source = component["engines"][0]
            with np.load(capture, allow_pickle=False) as arrays:
                for trial in component["trials"]:
                    seed = trial["seed"]
                    vector = arrays[f"{source}__{seed}__parameters"]
                    if vector.size != sum(value.numel() for value in parameters.values()):
                        raise ValueError("Captured parameter vector has the wrong length")
                    offset = 0
                    with torch.no_grad():
                        for value in parameters.values():
                            size = value.numel()
                            value.copy_(
                                torch.from_numpy(
                                    vector[offset : offset + size].reshape(value.shape)
                                ).to(value.device)
                            )
                            offset += size
                    validation = _evaluate(model, evaluation, engine)
                    model.zero_grad(set_to_none=True)
                    indices = _schedule(
                        seed, len(train), settings["steps"], microbatch, accumulation
                    )[-1]
                    batches = [_batch(tok, [train[i] for i in group]) for group in indices]
                    count = sum(
                        int(batch["labels"][:, 1:].ne(-100).sum()) for batch in batches
                    )
                    loss = 0.0
                    for batch in batches:
                        item = _loss(model, batch, count, engine)
                        loss += float(item.detach())
                        item.backward()
                    gradient = (
                        torch.cat(
                            [value.grad.detach().reshape(-1) for value in parameters.values()]
                        )
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    if not np.isfinite(gradient).all() or not np.isfinite(loss):
                        raise ValueError("Nonfinite shared-state loss or gradient")
                    key = f"{source}__{seed}__gradient"
                    gradients[key] = gradient
                    states.append(
                        {
                            "source_engine": source,
                            "seed": seed,
                            "parameter_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                            "validation_loss": validation,
                            "source_validation_loss": trial["final_validation_loss"],
                            "validation_loss_gap_from_source": validation
                            - trial["final_validation_loss"],
                            "gradient_batch_loss": loss,
                            "gradient_key": key,
                        }
                    )
                    print(
                        json.dumps({"engine": engine, **states[-1]}),
                        file=sys.stderr,
                        flush=True,
                    )
    if _frozen(model, engine) != frozen:
        raise ValueError("Replay changed frozen weights")
    with open(capture_path, "wb") as stream:
        np.savez(stream, **gradients)
    with open(capture_path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "engine": engine,
        "settings": settings,
        "states": states,
        "frozen_unchanged": True,
        "capture": {"sha256": digest, "parameter_order": expected_order},
        "source_captures": [
            component["capture"]["sha256"] for component in comparison["components"]
        ],
        "gradient_scope": "Post-update captured final adapters; last scheduled update batch, without a parameter update",
        "ort_optimizer": "disabled",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--captures", nargs="+", required=True)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--library")
    parser.add_argument("--corpus", help="The text file used by the captured training runs")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replay(
        args.reports,
        args.captures,
        engine=args.engine,
        library=args.library,
        capture_path=args.capture,
        corpus_path=args.corpus,
    )
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
