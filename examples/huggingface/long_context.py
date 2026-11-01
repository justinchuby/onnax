# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run one isolated long-context Qwen SFT qualification process."""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import subprocess
import time


def memory_limit(library, limit):
    """Cap the provider's linked MLX allocator before constructing sessions."""
    api = ctypes.CDLL(library)
    api.mlx_set_memory_limit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    api.mlx_set_memory_limit.restype = ctypes.c_int
    previous = ctypes.c_size_t()
    if api.mlx_set_memory_limit(ctypes.byref(previous), limit) != 0:
        raise RuntimeError("Could not set the MLX allocator limit")
    return api, previous.value


def _require_decrease(losses):
    if not losses[1] < losses[0]:
        raise AssertionError("Fixed training-set loss did not decrease")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--checkpoint-layers", type=int, default=1)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        parser.error("Select a new output directory")
    if os.environ.get("ONNXRUNTIME_EP_MLX_NO_COMPILE") != "1":
        parser.error("Set ONNXRUNTIME_EP_MLX_NO_COMPILE=1 for qualification")
    os.environ.setdefault("ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES", "536870912")
    api, previous = memory_limit(args.library, 20 * 1024**3)

    import torch
    from transformers import TrainerCallback

    from examples.huggingface.sft import run
    from examples.process_memory import physical_footprint

    with open(os.path.join(args.corpus, "manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    for metadata in manifest["files"].values():
        with open(os.path.join(args.corpus, metadata["filename"]), "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != metadata["sha256"]:
            raise ValueError("Corpus file digest differs from its manifest")
    os.makedirs(args.output_dir)
    journal = {
        "status": "running",
        "precision": args.precision,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "versions": {},
        "mlx_memory_limit_bytes": 20 * 1024**3,
        "previous_mlx_memory_limit_bytes": previous,
        "mlx_cache_limit_bytes": int(os.environ["ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES"]),
        "updates": [],
    }
    from importlib.metadata import version

    journal["versions"] = {
        name: version(name)
        for name in (
            "onnx",
            "onnx-ir",
            "onnxscript",
            "onnxruntime",
            "numpy",
            "torch",
            "transformers",
            "trl",
            "peft",
        )
    }
    with open(args.library, "rb") as stream:
        journal["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    for name in ("active", "cache", "peak"):
        fn = getattr(api, f"mlx_get_{name}_memory")
        fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int

    def persist():
        path = os.path.join(args.output_dir, "qualification.json")
        with open(path + ".tmp", "w") as stream:
            json.dump(journal, stream, indent=2, allow_nan=False)
        os.replace(path + ".tmp", path)

    class Observe(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
            for parameter in model.onnx.adapters:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Missing or nonfinite adapter gradient")
                if parameter.grad.dtype != torch.float32:
                    raise AssertionError("Adapter gradients must accumulate in FP32")

        def on_step_end(self, args, state, control, model, optimizer, **kwargs):
            for parameter in model.onnx.adapters:
                if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
                    raise FloatingPointError("Invalid FP32 adapter update")
            for values in optimizer.state.values():
                for name in ("exp_avg", "exp_avg_sq"):
                    value = values[name]
                    if value.dtype != torch.float32 or not torch.isfinite(value).all():
                        raise FloatingPointError("Invalid FP32 AdamW state")
            row = {"step": state.global_step, "elapsed_seconds": time.perf_counter() - start}
            row["physical_footprint"] = physical_footprint()
            for name in ("active", "cache", "peak"):
                value = ctypes.c_size_t()
                if getattr(api, f"mlx_get_{name}_memory")(ctypes.byref(value)) != 0:
                    raise RuntimeError("Could not read MLX memory")
                row[f"mlx_{name}_bytes"] = value.value
            journal["updates"].append(row)
            persist()
            print(json.dumps(row), flush=True)

    with open(os.path.join(args.corpus, "manifest.json"), "rb") as stream:
        journal["corpus_manifest_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    start = time.perf_counter()
    persist()
    try:
        report = run(
            args.output_dir,
            real=True,
            steps=args.steps,
            resume=args.resume,
            mode="chunked",
            chunk_size=128,
            nf4=True,
            provider="mlx",
            library=args.library,
            projection_precision=args.precision,
            checkpoint_layers=args.checkpoint_layers,
            train_file=os.path.join(args.corpus, "train.jsonl"),
            validation_file=os.path.join(args.corpus, "validation.jsonl"),
            max_length=args.max_length,
            evaluate_train=True,
            callbacks=[Observe()],
        )
        losses = report["fixed_training_loss"]
        journal["fixed_training_loss_decreased"] = losses[1] < losses[0]
        _require_decrease(losses)
        journal["status"] = "complete"
    except BaseException as error:
        journal.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        journal["elapsed_seconds"] = time.perf_counter() - start
        journal["physical_footprint"] = physical_footprint()
        journal["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        persist()


if __name__ == "__main__":
    main()
