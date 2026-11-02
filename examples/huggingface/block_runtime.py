# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Capture real decoder pullback inputs and replay one block in isolation."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
import onnx_ir as ir

from examples.process_memory import physical_footprint

try:
    import resource
except ImportError:
    resource = None


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arrays_record(values):
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": value.nbytes,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in values.items()
    }


def check_outputs(names, values, expected):
    checks = {}
    for key, value in zip(names, values, strict=True):
        target = expected[key]
        if (
            value.dtype != target.dtype
            or value.shape != target.shape
            or not np.isfinite(value).all()
        ):
            raise AssertionError(f"Invalid replay gradient: {key}")
        np.testing.assert_allclose(value, target, rtol=1e-4, atol=1e-6)
        checks[key] = {
            "max_abs": float(np.max(np.abs(value - target))) if value.size else 0.0,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    return checks


def interface(directory):
    with open(os.path.join(directory, "interface.json"), encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("version") != 1 or result.get("kind") != "checkpointed_vjp":
        raise ValueError("Expected a version 1 checkpointed execution bundle")
    return result


def block_rows(record, names):
    rows = {row["name"]: (index, row) for index, row in enumerate(record["blocks"])}
    if not names or len(rows) != len(record["blocks"]) or len(set(names)) != len(names):
        raise ValueError("Block names must be unique")
    if any(name not in rows or not rows[name][1]["backward"] for name in names):
        raise ValueError("Select differentiable block names from the bundle interface")
    return {name: rows[name] for name in names}


class MlxMemory:
    """Read the same native allocator used by the selected EP library."""

    def __init__(self, library):
        self.api = ctypes.CDLL(library)
        for name in ("active", "cache", "peak"):
            fn = getattr(self.api, f"mlx_get_{name}_memory")
            fn.argtypes, fn.restype = [ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int
        self.api.mlx_reset_peak_memory.argtypes = []
        self.api.mlx_reset_peak_memory.restype = ctypes.c_int

    def snapshot(self):
        result = {}
        for name in ("active", "cache", "peak"):
            value = ctypes.c_size_t()
            if getattr(self.api, f"mlx_get_{name}_memory")(ctypes.byref(value)) != 0:
                raise RuntimeError("Could not read MLX allocator counters")
            result[name + "_bytes"] = value.value
        return result

    def reset_peak(self):
        if self.api.mlx_reset_peak_memory() != 0:
            raise RuntimeError("Could not reset MLX allocator peak")


def capture(export, *, names, batch, output, factory):
    """Record actual per-block arguments/results during one task backward."""
    from examples.huggingface.decoder_sft import OnnxDecoder

    directory = os.path.join(export, "execution")
    record = interface(directory)
    selected = block_rows(record, names)
    targets = {
        f"block_{index}_backward.onnx": (name, row, index)
        for name, (index, row) in selected.items()
    }
    os.makedirs(output, exist_ok=False)
    captures = {}

    class RecordingSession:
        def __init__(self, source):
            self.original = factory(source)
            self.filename = os.path.basename(os.fspath(source))

        def __getattr__(self, name):
            return getattr(self.original, name)

        def run(self, outputs, feeds):
            values = self.original.run(outputs, feeds)
            if self.filename in targets:
                name, row, index = targets[self.filename]
                if name in captures or list(outputs) != row["gradients"]:
                    raise ValueError("Capture expects one backward call with declared outputs")
                results = dict(zip(outputs, values, strict=True))
                if any(not np.isfinite(value).all() for value in results.values()):
                    raise FloatingPointError("Nonfinite captured block gradient")
                feed_file, result_file = (
                    f"block_{index}_feeds.npz",
                    f"block_{index}_outputs.npz",
                )
                np.savez(os.path.join(output, feed_file), **feeds)
                np.savez(os.path.join(output, result_file), **results)
                captures[name] = {
                    "index": index,
                    "model": self.filename,
                    "feeds": feed_file,
                    "outputs": result_file,
                    "input_arrays": arrays_record(feeds),
                    "output_arrays": arrays_record(results),
                    "output_names": list(outputs),
                }
            return values

    model = OnnxDecoder.load(export, session_factory=RecordingSession)
    try:
        result = model(**batch)
        result.loss.backward()
        if set(captures) != set(names):
            raise ValueError("Selected block did not participate in backward")
        result = {
            "version": 1,
            "bundle_identity": record["identity"],
            "interface_sha256": digest(os.path.join(directory, "interface.json")),
            "batch": arrays_record(
                {name: value.detach().numpy() for name, value in batch.items()}
            ),
            "loss": float(result.loss.detach()),
            "blocks": captures,
            "pretrained_source": getattr(model.config, "onnax_pretrained", None),
        }
        with open(os.path.join(output, "capture.json"), "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        return result
    finally:
        model.close()


def load_case(directory, captures, name):
    """Verify a bundle and return a recorded block and its immutable arguments."""
    with open(os.path.join(captures, "capture.json"), encoding="utf-8") as stream:
        captured = json.load(stream)
    record = interface(directory)
    selected = block_rows(record, [name])[name]
    if (
        captured["version"] != 1
        or captured["bundle_identity"] != record["identity"]
        or captured["interface_sha256"] != digest(os.path.join(directory, "interface.json"))
    ):
        raise ValueError("Captured inputs belong to a different execution bundle")
    for filename, expected in record["files"].items():
        if digest(os.path.join(directory, filename)) != expected:
            raise ValueError(f"Bundle file integrity mismatch: {filename}")
    block = captured["blocks"][name]
    if (
        block["index"] != selected[0]
        or block["model"] != f"block_{selected[0]}_backward.onnx"
        or block["output_names"] != selected[1]["gradients"]
    ):
        raise ValueError("Captured block interface differs from the execution bundle")
    with np.load(os.path.join(captures, block["feeds"]), allow_pickle=False) as values:
        feeds = {key: values[key] for key in values.files}
    with np.load(os.path.join(captures, block["outputs"]), allow_pickle=False) as values:
        expected = {key: values[key] for key in values.files}
    if (
        arrays_record(feeds) != block["input_arrays"]
        or arrays_record(expected) != block["output_arrays"]
    ):
        raise ValueError("Captured arrays differ from their recorded hashes")
    return record, block, feeds, expected


def measure(directory, captures, name, *, factory, memory=None, warmup=10, samples=100):
    """Replay a captured checkpoint derivative; include runtime boundary exchange."""
    if warmup < 0 or samples < 1:
        raise ValueError("Expected warmup >= 0 and samples >= 1")
    record, block, feeds, expected = load_case(directory, captures, name)
    selected = record["blocks"][block["index"]]
    model = ir.load(os.path.join(directory, block["model"]))
    initializer_bytes = sum(
        value.const_value.nbytes for value in model.graph.initializers.values()
    )
    before = time.perf_counter()
    executor = factory(os.path.join(directory, block["model"]))
    construction = time.perf_counter() - before
    if {value.name for value in executor.get_inputs()} != set(feeds):
        raise ValueError("Captured feed names differ from runtime inputs")
    before = time.perf_counter()
    outputs = executor.run(block["output_names"], feeds)
    first_execution = time.perf_counter() - before
    checks = check_outputs(block["output_names"], outputs, expected)
    del outputs
    for _ in range(warmup):
        executor.run(block["output_names"], feeds)
    if memory is not None:
        memory.reset_peak()
    allocator_before = None if memory is None else memory.snapshot()
    durations = []
    for index in range(samples):
        before = time.perf_counter()
        outputs = executor.run(block["output_names"], feeds)
        durations.append(time.perf_counter() - before)
        if index == samples - 1:
            final_checks = check_outputs(block["output_names"], outputs, expected)
        del outputs
    allocator_after = None if memory is None else memory.snapshot()
    window_peak = (
        None
        if memory is None
        else max(allocator_before["active_bytes"], allocator_after["peak_bytes"])
    )
    if arrays_record(feeds) != block["input_arrays"]:
        raise AssertionError("Runtime mutated a captured input")
    peak = None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "version": 1,
        "block": name,
        "bundle_identity": record["identity"],
        "executed_model_sha256": record["files"][block["model"]],
        "input_arrays": block["input_arrays"],
        "output_arrays": block["output_arrays"],
        "dynamic_input_bytes": sum(value.nbytes for value in feeds.values()),
        "dynamic_output_bytes": sum(value.nbytes for value in expected.values()),
        "trainable_input_bytes": sum(
            value.nbytes
            for key, value in feeds.items()
            if key in record["specification"]["parameters"]
        ),
        "gradient_mapping": dict(
            zip(selected["active_inputs"], selected["gradients"], strict=True)
        ),
        "main_graph_nodes": len(model.graph),
        "onnx_file_bytes": os.path.getsize(os.path.join(directory, block["model"])),
        "declared_initializer_bytes": initializer_bytes,
        "outputs": checks,
        "final_outputs": final_checks,
        "session_creation_seconds": construction,
        "first_execution_seconds": first_execution,
        "warmup": warmup,
        "samples": durations,
        "median_seconds": float(np.median(durations)),
        "p95_seconds": float(np.percentile(durations, 95)),
        "allocator_before": allocator_before,
        "allocator_after": allocator_after,
        "window_allocator_peak_bytes": window_peak,
        "incremental_allocator_peak_bytes": None
        if memory is None
        else window_peak - allocator_before["active_bytes"],
        "process_peak_rss_bytes": None
        if peak is None
        else peak * (1 if sys.platform == "darwin" else 1024),
        "physical_memory": physical_footprint(),
        "memory_scope": "Whole worker including imports, bundle verification, one derivative session and captured CPU arrays; allocator peak reset after qualification/warmup",
        "timing_scope": "One exported block recomputation/VJP session, including runtime host/device exchange; fixed parameters; excludes other blocks, task loss and optimizer",
        "ort_graph_optimization": "disabled",
        "cpu_threads": 1,
        "platform": platform.platform(),
        "tool_source_sha256": digest(__file__),
        "versions": {key: version(key) for key in ("numpy", "onnxruntime")},
    }


def load_batch(path, *, vocab_size, max_sequence):
    """Load a recorded training microbatch without tokenization or truncation."""
    import torch

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"input_ids", "attention_mask", "position_ids", "labels"}
    if arrays.keys() != required:
        raise ValueError("Batch requires input_ids, attention_mask, position_ids and labels")
    ids = arrays["input_ids"]
    if ids.ndim != 2 or ids.shape[0] < 1 or not 2 <= ids.shape[1] <= max_sequence:
        raise ValueError("Expected a nonempty rank-2 batch within the model context limit")
    if any(value.dtype != np.int64 or value.shape != ids.shape for value in arrays.values()):
        raise ValueError("Batch arrays must have matching shapes and int64 dtype")
    labels, mask, positions = (
        arrays[key] for key in ("labels", "attention_mask", "position_ids")
    )
    if np.any((ids < 0) | (ids >= vocab_size)) or np.any(
        (labels != -100) & ((labels < 0) | (labels >= vocab_size))
    ):
        raise ValueError("Token IDs and valid labels must be within the vocabulary")
    if np.any((mask != 0) & (mask != 1)) or np.any((mask == 0) & (labels != -100)):
        raise ValueError("Padding requires a binary mask and ignored labels")
    if np.any((positions < 0) | (positions >= max_sequence)):
        raise ValueError("Position IDs must be within the model position limit")
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("capture")
    get.add_argument("--export", required=True)
    get.add_argument("--block", action="append", required=True)
    batch_options = get.add_mutually_exclusive_group()
    batch_options.add_argument("--sequence", type=int)
    batch_options.add_argument("--batch-file")
    get.add_argument("--cache-dir")
    get.add_argument("--output", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--bundle", required=True)
    run.add_argument("--captures", required=True)
    run.add_argument("--block", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--samples", type=int, default=100)
    run.add_argument("--unfused", action="store_true")
    run.add_argument("--output", required=True)
    for child in (get, run):
        child.add_argument("--provider", choices=["cpu", "mlx"], default="cpu")
        child.add_argument("--library")
    args = parser.parse_args()
    if args.provider == "mlx":
        if not args.library:
            parser.error("MLX measurements require an explicit --library")
        os.environ["ONNXRUNTIME_EP_MLX_NO_COMPILE"] = "1"
        os.environ["ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES"] = "536870912"
        os.environ["ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE"] = str(
            int(getattr(args, "unfused", False))
        )
        from examples.lora.mlx_runtime import register_library, session

        register_library(args.library)
        memory = MlxMemory(args.library)
    else:
        if args.library or getattr(args, "unfused", False):
            parser.error("Library and fusion controls require the MLX provider")
        from examples.lora.train import session

        memory = None
    if args.command == "capture":
        import torch
        from transformers import AutoConfig

        from examples.huggingface.pretrained import PretrainedSource, load_tokenizer
        from examples.huggingface.sft import tokenizer

        torch.set_num_threads(1)
        config = AutoConfig.from_pretrained(args.export, local_files_only=True)
        if args.batch_file:
            batch = load_batch(
                args.batch_file,
                vocab_size=config.vocab_size,
                max_sequence=config.max_position_embeddings,
            )
        else:
            args.sequence = 64 if args.sequence is None else args.sequence
            if args.sequence < 2:
                parser.error("Expected sequence >= 2")
            if args.sequence > config.max_position_embeddings:
                parser.error("Sequence exceeds the model position limit")
            source = getattr(config, "onnax_pretrained", None)
            tok = (
                tokenizer()
                if source is None
                else load_tokenizer(
                    PretrainedSource(source["model_id"], source["revision"]),
                    cache_dir=args.cache_dir,
                )
            )
            text = "ONNX models learn from examples. Low-rank adapters update a small set of parameters. "
            repetitions = 1
            ids = []
            while len(ids) < args.sequence:
                ids = tok.encode(text * repetitions, add_special_tokens=False)
                repetitions *= 2
            ids = torch.tensor([ids[: args.sequence]], dtype=torch.int64)
            batch = {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "position_ids": torch.arange(args.sequence)[None],
                "labels": ids.clone(),
            }
        result = capture(
            args.export, names=args.block, batch=batch, output=args.output, factory=session
        )
    else:
        result = measure(
            args.bundle,
            args.captures,
            args.block,
            factory=session,
            memory=memory,
            warmup=args.warmup,
            samples=args.samples,
        )
        result.update(
            provider=args.provider,
            cpu_fallback_disabled=args.provider == "mlx",
            library_sha256=None if args.library is None else digest(args.library),
            gated_delta_reverse_fused=None if args.provider != "mlx" else not args.unfused,
            runtime_environment={
                key: os.environ.get(key)
                for key in (
                    "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                    "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                )
            },
        )
        if args.provider == "mlx":
            result["versions"]["mlx"] = version("mlx")
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                "command": args.command,
                "block": getattr(args, "block", None),
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
